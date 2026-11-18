"""Two-sided statistics of the weight matrices being merged.

For every task t and weight matrix W_l (an ``nn.Linear``), IF-Merge needs

* ``A[l]``: the second moment of the inputs of W_l (activation statistics), and
* ``G[l]``: the second moment of the feedback that reaches the output of W_l, i.e. of
  the gradient of a prediction loss with respect to that output.

Both are accumulated with module hooks during ordinary forward and backward passes; a
backend only has to run its model and call ``backward()`` on its feedback loss.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# The first layer's input needs no gradient; its output feedback is still what we record.
warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed")


class SecondMoment:
    """Running sum of x x^T over the rows of the samples it is shown."""

    def __init__(self) -> None:
        self.count = 0
        self.gram: torch.Tensor | None = None
        self._fourth_moment = 0.0  # sum of ||x||^4, the only extra term Ledoit-Wolf needs

    def update(self, samples: torch.Tensor) -> None:
        rows = samples.reshape(-1, samples.shape[-1]).float()
        gram = rows.transpose(0, 1) @ rows
        self.gram = gram if self.gram is None else self.gram.add_(gram)
        self.count += rows.shape[0]
        self._fourth_moment += float(rows.square().sum(dim=-1).square().sum())

    def mean(self, count: int | None = None) -> torch.Tensor:
        assert self.gram is not None
        return self.gram / max(self.count if count is None else count, 1)

    def ledoit_wolf_shrinkage(self) -> float:
        """Optimal shrinkage toward (tr(S)/d) I for the uncentered second moment S."""
        if self.gram is None or self.count < 2:
            return 1.0
        dimension = self.gram.shape[0]
        moment = self.gram / self.count
        isotropic = float(moment.diagonal().mean())
        identity = torch.eye(dimension, device=moment.device, dtype=moment.dtype)
        spread = float((moment - isotropic * identity).square().sum()) / dimension
        if spread <= 0:
            return 1.0
        noise = (self._fourth_moment - self.count * float(moment.square().sum())) / (
            self.count**2 * dimension
        )
        return float(min(max(noise, 0.0), spread) / spread)


@dataclass
class TaskStatistics:
    """Statistics of one task, measured at one model. Tensors live on the CPU."""

    A: dict[str, torch.Tensor] = field(default_factory=dict)
    G: dict[str, torch.Tensor] = field(default_factory=dict)
    G_shrinkage: dict[str, float] = field(default_factory=dict)  # Ledoit-Wolf estimates


class StatisticsCollector:
    """Hooks that accumulate input and output-gradient second moments of linear layers.

    >>> collector = StatisticsCollector(layers)
    >>> with collector:
    ...     loss = feedback_loss(model(batch))   # forward pass: inputs are recorded
    ...     loss.sum().backward()                # backward pass: feedback is recorded
    >>> statistics = collector.finish()
    """

    def __init__(
        self,
        layers: Mapping[str, nn.Module],
        feedback: bool = True,
        output_activations: bool = False,
        rows: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
        inputs_only: Collection[str] = (),
    ) -> None:
        """
        ``output_activations`` records the layer outputs themselves in place of the feedback.
        ``rows(layer, tensor)`` selects what is accumulated, e.g. the valid positions of a
        padded batch. Layers in ``inputs_only`` get no output-side statistic.
        """
        self.layers = dict(layers)
        self.feedback = feedback and not output_activations
        self.output_activations = output_activations
        self.rows = rows or (lambda _name, tensor: tensor)
        self.inputs = {name: SecondMoment() for name in self.layers}
        self.outputs = (
            {name: SecondMoment() for name in self.layers if name not in inputs_only}
            if feedback or output_activations
            else {}
        )
        self._handles: list = []

    def __enter__(self) -> StatisticsCollector:
        for name, layer in self.layers.items():
            self._handles.append(layer.register_forward_hook(self._record_input(name)))
            if self.feedback and name in self.outputs:
                self._handles.append(layer.register_full_backward_hook(self._record_feedback(name)))
        return self

    def __exit__(self, *_) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _record_input(self, name: str):
        def hook(_module, inputs, output):
            self.inputs[name].update(self.rows(name, inputs[0].detach()))
            if self.output_activations and name in self.outputs:
                self.outputs[name].update(self.rows(name, output.detach()))

        return hook

    def _record_feedback(self, name: str):
        def hook(_module, _grad_input, grad_output):
            if grad_output[0] is not None:
                self.outputs[name].update(self.rows(name, grad_output[0].detach()))

        return hook

    def finish(self, backward_passes_per_sample: int = 1) -> TaskStatistics:
        """Average the accumulated moments.

        ``backward_passes_per_sample`` is larger than one for estimators that
        backpropagate several times through the same forward pass (one pass per class
        for the exact Fisher); the feedback moment is then a sum over those passes.
        """
        statistics = TaskStatistics()
        for name, moment in self.inputs.items():
            if moment.gram is None:
                continue
            statistics.A[name] = moment.mean().cpu()
        for name, moment in self.outputs.items():
            if moment.gram is None:
                continue
            feedback = moment.mean(moment.count // backward_passes_per_sample)
            statistics.G[name] = feedback.cpu()
            inactive = not bool(torch.count_nonzero(feedback))
            statistics.G_shrinkage[name] = 0.0 if inactive else moment.ledoit_wolf_shrinkage()
        return statistics


def train_only(model: nn.Module, layers: Iterable[nn.Module]) -> None:
    """Enable gradients for the weights of ``layers`` only, and clear old gradients."""
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    for layer in layers:
        layer.weight.requires_grad_(True)


def combine_statistics(
    fresh: list[TaskStatistics], first: list[TaskStatistics] | None, reestimate: str
) -> list[TaskStatistics]:
    """Keep the first-round value of every factor that is not re-estimated."""
    if first is None or reestimate == "both":
        return fresh
    if reestimate == "none":
        return first
    combined = []
    for new, old in zip(fresh, first, strict=True):
        if reestimate == "a":  # A is measured again, G stays at its first-round value
            combined.append(TaskStatistics(new.A, old.G, old.G_shrinkage))
        else:
            combined.append(TaskStatistics(old.A, new.G, new.G_shrinkage))
    return combined


def adjust_feedback(
    statistics: list[TaskStatistics], permute_tasks: bool = False, heterogeneity: float = 1.0
) -> list[TaskStatistics]:
    """Controls on the task correspondence of G (ablations).

    ``permute_tasks`` gives task t the G of task t + 1 (cyclically). ``heterogeneity`` = eta
    replaces the scale-normalized G_t by eta G_t + (1 - eta) mean_s G_s within each layer, and
    interpolates the shrinkage coefficients in the same way; tasks without feedback stay without.
    """
    if permute_tasks:
        shifted = statistics[1:] + statistics[:1]
        statistics = [
            TaskStatistics(task.A, other.G, other.G_shrinkage)
            for task, other in zip(statistics, shifted, strict=True)
        ]
    if heterogeneity != 1.0:
        for layer in statistics[0].G:
            normalized, active = [], []
            for index, task in enumerate(statistics):
                moment = task.G[layer].float()
                scale = moment.diagonal().mean()
                if torch.isfinite(scale) and float(scale.abs()) > 1e-30:
                    normalized.append(moment / scale)
                    active.append(index)
            if not active:
                continue
            mean = torch.stack(normalized).mean(dim=0)
            mean_shrinkage = sum(statistics[i].G_shrinkage[layer] for i in active) / len(active)
            for moment, index in zip(normalized, active, strict=True):
                task = statistics[index]
                task.G[layer] = heterogeneity * moment + (1.0 - heterogeneity) * mean
                task.G_shrinkage[layer] = (
                    heterogeneity * task.G_shrinkage[layer] + (1.0 - heterogeneity) * mean_shrinkage
                )
    return statistics
