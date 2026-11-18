"""AdaMerging: merging coefficients learned by entropy minimization on unlabeled data."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .regmean import State
from .weight_space import TaskVector


def adamerging(
    pretrained: State,
    task_vectors: Sequence[TaskVector],
    task_logits: Callable[[int, dict[str, torch.Tensor]], torch.Tensor],
    steps: int = 500,
    learning_rate: float = 1e-3,
    initial_coefficient: float = 0.3,
    device: str = "cuda",
    on_step: Callable[[int, float], None] | None = None,
) -> dict[str, torch.Tensor]:
    """Layer-wise AdaMerging: one coefficient per task and per tensor of the state dict.

    The merged model is ``theta_0 + sum_t lambda_t[tensor] * tau_t[tensor]`` with the
    coefficients clamped to [0, 1]. They are trained with Adam to minimize the entropy of
    the merged model's predictions, summed over tasks. ``task_logits(t, state)`` draws a
    batch of unlabeled inputs of task t and returns the logits of the model with
    parameters ``state`` (e.g. through ``torch.func.functional_call``), differentiable
    with respect to ``state``. Returns the merged state after the last step, on ``device``.

    Yang et al., "AdaMerging: Adaptive Model Merging for Multi-Task Learning", ICLR 2024.
    """
    task_count = len(task_vectors)
    names = [name for name, tensor in pretrained.items() if tensor.is_floating_point()]
    base = {name: tensor.to(device).float() for name, tensor in pretrained.items()}
    stacked = {name: torch.stack([tv[name].to(device) for tv in task_vectors]) for name in names}
    coefficients = torch.full(
        (task_count, len(names)), initial_coefficient, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([coefficients], lr=learning_rate)

    def merged_state() -> dict[str, torch.Tensor]:
        clamped = coefficients.clamp(0.0, 1.0)
        state = dict(base)
        for column, name in enumerate(names):
            state[name] = base[name] + torch.tensordot(clamped[:, column], stacked[name], dims=([0], [0]))
        return state

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        state = merged_state()
        total = 0.0
        for task in range(task_count):
            log_probabilities = task_logits(task, state).log_softmax(dim=-1)
            entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1).mean()
            entropy.backward(retain_graph=task < task_count - 1)  # keep the graph of the merged weights
            total += entropy.item()
        optimizer.step()
        if on_step:
            on_step(step, total / task_count)

    with torch.no_grad():
        return {name: tensor.detach() for name, tensor in merged_state().items()}
