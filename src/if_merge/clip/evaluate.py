"""Classification with a CLIP visual encoder and frozen zero-shot heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from .assets import ClipAssets, Head, normalize_images
from .model import State, build_exposed_visual


def head_to(head: Head, device: str | torch.device) -> Head:
    weight, bias, normalize = head
    return weight.to(device).float(), None if bias is None else bias.to(device).float(), normalize


def head_logits(features: torch.Tensor, head: Head) -> torch.Tensor:
    """Logits of visual features under a head that already lives on their device."""
    weight, bias, normalize = head
    features = features.float()
    if normalize:
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    logits = features @ weight.transpose(0, 1)
    return logits if bias is None else logits + bias


def classify(visual: nn.Module, images: torch.Tensor, head: Head) -> torch.Tensor:
    """Logits of normalized ``images``."""
    return head_logits(visual(images), head)


@torch.no_grad()
def accuracy(
    visual: nn.Module,
    head: Head,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    batch_size: int = 1024,
) -> float:
    """Top-1 accuracy (%) on a uint8 image cache, with float16 autocast on CUDA."""
    visual = visual.to(device).eval()
    head = head_to(head, device)
    correct = 0
    for start in range(0, len(labels), batch_size):
        batch = normalize_images(images[start : start + batch_size], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            features = visual(batch)
        logits = head_logits(features, head)  # the head stays in float32
        correct += int((logits.argmax(dim=-1) == labels[start : start + batch_size].to(device)).sum())
    return 100.0 * correct / len(labels)


def evaluate(
    architecture: str,
    state: State,
    assets: ClipAssets,
    tasks: Sequence[str],
    device: str,
    split: str = "test",
    heads: Mapping[str, Head] | None = None,
) -> dict:
    """Per-task and task-averaged accuracy of a visual state dict on ``split``."""
    visual = build_exposed_visual(architecture, state).to(device)
    per_task = {}
    for task in tasks:
        images, labels = assets.load_images(task, split)
        head = heads[task] if heads else assets.load_head(task)
        per_task[task] = accuracy(visual, head, images, labels, device)
        del images, labels
    return {"average": sum(per_task.values()) / len(per_task), "per_task": per_task}


class DiagnosticKL:
    """Expert-merged KL on labeled-split images that the method never reads.

    The first ``images`` validation images of every task; the experts' predictions are computed
    once. Used to check that a lower holdout KL is not an artifact of selecting on the holdout set.
    """

    def __init__(
        self, architecture: str, assets: ClipAssets, tasks: Sequence[str], device: str, images: int = 256
    ):
        self.architecture, self.tasks, self.device = architecture, tuple(tasks), device
        self.heads = {task: head_to(assets.load_head(task), device) for task in self.tasks}
        self.images = {task: assets.load_images(task, "val")[0][:images] for task in self.tasks}
        self.teachers = {
            task: self._logits(build_exposed_visual(architecture, assets.load_checkpoint(task)), task).cpu()
            for task in self.tasks
        }

    @torch.no_grad()
    def _logits(self, visual: nn.Module, task: str, batch_size: int = 128) -> torch.Tensor:
        visual, chunks = visual.to(self.device).eval(), []
        for start in range(0, len(self.images[task]), batch_size):
            batch = normalize_images(self.images[task][start : start + batch_size], self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
                features = visual(batch)
            chunks.append(head_logits(features, self.heads[task]))
        return torch.cat(chunks)

    @torch.no_grad()
    def __call__(self, state: State) -> dict:
        visual = build_exposed_visual(self.architecture, state)
        per_task = {}
        for task in self.tasks:
            teacher = self.teachers[task].to(self.device).log_softmax(dim=-1)
            student = self._logits(visual, task).log_softmax(dim=-1)
            per_task[task] = float((teacher.exp() * (teacher - student)).sum()) / len(teacher)
        return {"average": sum(per_task.values()) / len(per_task), "per_task": per_task}
