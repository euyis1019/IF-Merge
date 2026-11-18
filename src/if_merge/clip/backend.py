"""IF-Merge backend for CLIP visual encoders on classification tasks.

Each image has one classification distribution over the classes of its task, given by
the frozen zero-shot head. The expert of each task is the teacher; its logits on the
calibration images are computed once and cached.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ..config import IFMergeConfig
from ..core.feedback import feedback_loss
from ..core.statistics import StatisticsCollector, TaskStatistics, train_only
from .assets import ClipAssets, calibration_split, normalize_images
from .evaluate import classify, head_to
from .model import (
    State,
    build_exposed_visual,
    expose_linear_maps,
    linear_layers,
    open_clip_state,
    transformer_blocks,
)


class ClipBackend:
    """Owns the merged visual encoder, the experts' weights and the calibration data."""

    activation_only: frozenset[str] = frozenset()

    def __init__(
        self,
        architecture: str,
        tasks: Sequence[str],
        assets: ClipAssets,
        initial_state: State,
        config: IFMergeConfig,
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        self.tasks = tuple(tasks)
        self.device = device
        self.batch_size = batch_size
        self.samples = config.calibration.samples
        self.holdout = config.calibration.holdout
        self.seed = config.calibration.seed

        self.visual = build_exposed_visual(architecture, initial_state).to(device)
        self.layers = linear_layers(self.visual)
        self.blocks = transformer_blocks(self.layers)
        self.heads = {task: head_to(assets.load_head(task), device) for task in self.tasks}

        # Unlabeled calibration images: C_t for the statistics, H_t for step selection.
        total = self.samples + self.holdout
        statistics_index, holdout_index = calibration_split(self.samples, self.holdout)
        self.calibration, self.heldout = {}, {}
        self._expert_weights: dict[str, list[torch.Tensor]] = {name: [] for name in self.layers}
        for task in self.tasks:
            images, labels = assets.load_images(task, "calib", self.seed)
            if len(images) < total:
                raise ValueError(f"{task}: {len(images)} calibration images, {total} requested")
            images, labels = images[:total], labels[:total]
            expert = build_exposed_visual(architecture, assets.load_checkpoint(task), boundary=False)
            teacher = self._logits(expert.to(device), task, images)
            self.calibration[task] = tuple(x[statistics_index] for x in (images, teacher, labels))
            self.heldout[task] = (images[holdout_index], teacher[holdout_index])
            for name, layer in linear_layers(expose_linear_maps(expert)).items():
                self._expert_weights[name].append(layer.weight.detach().float().cpu().clone())
            del expert
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ protocol
    def expert_weights(self, layer: str) -> list[torch.Tensor]:
        return self._expert_weights[layer]

    def state(self) -> dict[str, torch.Tensor]:
        """The merged visual encoder as an open_clip visual state dict."""
        return open_clip_state(self.visual)

    @torch.no_grad()
    def holdout_kl(self) -> float:
        per_task = []
        for task in self.tasks:
            images, teacher = self.heldout[task]
            teacher = teacher.to(self.device)
            logits = self._logits(self.visual, task, images, to_cpu=False)
            kl = F.kl_div(logits.log_softmax(dim=-1), teacher.softmax(dim=-1), reduction="sum")
            per_task.append(float(kl) / len(images))
        return sum(per_task) / len(per_task)

    def collect_statistics(self, config: IFMergeConfig, round_index: int) -> list[TaskStatistics]:
        estimator = config.g.estimator
        train_only(self.visual, self.layers.values())
        generator = torch.Generator(device=self.device).manual_seed(self.seed + round_index)
        statistics = []
        for task in self.tasks:
            images, teacher, task_labels = self.calibration[task]
            collector = StatisticsCollector(
                self.layers,
                feedback=estimator != "identity",
                output_activations=estimator == "output_activation",
            )
            passes = 1
            with collector:
                for start in range(0, len(images), self.batch_size):
                    batch = slice(start, start + self.batch_size)
                    logits = classify(
                        self.visual, normalize_images(images[batch], self.device), self.heads[task]
                    )
                    teacher_batch = teacher[batch].to(self.device)
                    if estimator in ("normalized_kl", "kl"):
                        feedback_loss(teacher_batch, logits, estimator == "normalized_kl").sum().backward()
                    elif estimator == "mc_fisher":
                        probabilities = logits.detach().softmax(dim=-1)
                        for sample in range(config.g.mc_samples):
                            labels = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
                            F.cross_entropy(logits, labels, reduction="sum").backward(
                                retain_graph=sample < config.g.mc_samples - 1
                            )
                    elif estimator == "empirical_fisher":
                        labels = task_labels[batch].to(self.device).long()
                        F.cross_entropy(logits, labels, reduction="sum").backward()
                    elif estimator == "exact_fisher":
                        # sum_c q_c grad(log q_c) grad(log q_c)^T, one backward pass per class
                        weights = logits.detach().softmax(-1)
                        log_student = logits.log_softmax(dim=-1)
                        passes = logits.shape[-1]
                        for index in range(passes):
                            loss = -(weights[:, index].sqrt() * log_student[:, index]).sum()
                            loss.backward(retain_graph=index < passes - 1)
                    for parameter in self.visual.parameters():
                        parameter.grad = None
            statistics.append(collector.finish(passes))
            del collector
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        train_only(self.visual, ())
        return statistics

    # ------------------------------------------------------------- helpers
    @torch.no_grad()
    def _logits(self, visual, task: str, images: torch.Tensor, to_cpu: bool = True) -> torch.Tensor:
        batch_size = max(self.batch_size, 64)
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = normalize_images(images[start : start + batch_size], self.device)
            logits = classify(visual, batch, self.heads[task])
            chunks.append(logits.cpu() if to_cpu else logits)
        return torch.cat(chunks)
