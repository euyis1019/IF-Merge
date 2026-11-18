"""Files of the CLIP benchmark: expert checkpoints, classification heads and image caches.

Everything lives under one data root (``--data``, or the ``IF_MERGE_DATA`` environment
variable, default ``./data``), written by ``if-merge prepare-clip``::

    <data>/clip/
      checkpoints/<arch>/zeroshot.pt        pretrained CLIP        (open_clip state dict)
      checkpoints/<arch>/<Task>.pt          fine-tuned expert      (open_clip state dict)
      heads/<arch>/<Task>.pt                {"weight", "bias", "normalize"}
      images/<Task>/test.pt                 {"x": uint8 [N,3,224,224], "y": int64 [N]}
      images/<Task>/val.pt                  labeled split used to select baseline hyperparameters
      images/<Task>/train.pt                full training split (joint-training reference only)
      images/<Task>/calib_seed<K>.pt        unlabeled calibration draw K

Images are stored after the geometric part of the CLIP preprocessing (bicubic resize
to 224 and center crop) and before normalization, which happens on the GPU. The three
backbones share the caches because they share the input resolution.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..paths import default_data_root
from .tasks import ARCHITECTURES, TASKS

Head = tuple[torch.Tensor, torch.Tensor | None, bool]  # weight [classes, d], bias, normalize

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)


def budget_sizes(images: int) -> tuple[int, int]:
    """(|C_t|, |H_t|) for a budget of ``images`` unlabeled images per task.

    Up to 512 images the budget is halved (256 + 256 is the default). Larger budgets keep the
    256 holdout images of the default budget and give every further image to the statistics.
    """
    if images <= 512:
        if images < 2 or images % 2:
            raise ValueError("A budget of at most 512 images per task must be even.")
        return images // 2, images // 2
    return images - 256, 256


def calibration_split(samples: int, holdout: int) -> tuple[list[int], list[int]]:
    """Positions of C_t and H_t inside a calibration draw.

    C_t comes first and H_t follows it, except that H_t never starts beyond position 256:
    budgets larger than the default then share the default holdout set (positions 256-511)
    and are nested in one another.
    """
    first = min(samples, 256)
    return (
        list(range(first)) + list(range(first + holdout, samples + holdout)),
        list(range(first, first + holdout)),
    )


def normalize_images(images: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """uint8 image cache -> normalized float input of the visual encoder."""
    images = images.to(device).float().div_(255)
    return images.sub_(CLIP_MEAN.to(device)).div_(CLIP_STD.to(device))


class ClipAssets:
    def __init__(self, architecture: str, root: str | Path | None = None) -> None:
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture {architecture!r}; expected one of {ARCHITECTURES}")
        self.architecture = architecture
        self.root = Path(root) if root is not None else default_data_root()
        self.clip = self.root / "clip"

    # ---------------------------------------------------------------- paths
    def checkpoint_path(self, task: str | None) -> Path:
        name = "zeroshot" if task is None else task
        return self.clip / "checkpoints" / self.architecture / f"{name}.pt"

    def head_path(self, task: str) -> Path:
        return self.clip / "heads" / self.architecture / f"{task}.pt"

    def image_path(self, task: str, split: str, seed: int = 0) -> Path:
        name = f"calib_seed{seed}" if split == "calib" else split
        return self.clip / "images" / task / f"{name}.pt"

    def missing(self, tasks, splits=("calib", "test"), seed: int = 0) -> list[Path]:
        paths = [self.checkpoint_path(None)]
        for task in tasks:
            paths += [self.checkpoint_path(task), self.head_path(task)]
            paths += [self.image_path(task, split, seed) for split in splits]
        return [path for path in paths if not path.is_file()]

    # -------------------------------------------------------------- loaders
    def load_checkpoint(self, task: str | None) -> dict[str, torch.Tensor]:
        """Full CLIP state dict (``visual.*`` and text keys) of the pretrained model or an expert."""
        state = torch.load(self.checkpoint_path(task), map_location="cpu", weights_only=True)
        return {key.removeprefix("model."): value for key, value in state.items()}

    def load_head(self, task: str) -> Head:
        stored = torch.load(self.head_path(task), map_location="cpu", weights_only=True)
        weight, bias = stored["weight"], stored.get("bias")
        if weight.ndim != 2 or weight.shape[0] != TASKS[task].classes:
            raise ValueError(f"{task}: the classification head does not match the label space")
        return weight, bias, bool(stored["normalize"])

    def load_images(self, task: str, split: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """``split`` is "calib", "val", "test" or "train". Returns (uint8 images, labels)."""
        stored = torch.load(self.image_path(task, split, seed), map_location="cpu", weights_only=True)
        images, labels = stored["x"], stored["y"]
        if images.dtype != torch.uint8 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError(f"{task}/{split}: unexpected image cache")
        if len(images) != len(labels) or int(labels.max()) >= TASKS[task].classes:
            raise ValueError(f"{task}/{split}: labels do not match the images or the label space")
        return images, labels
