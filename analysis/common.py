"""Helpers shared by the analysis scripts: arguments, checkpoints, test subsets and logits.

Every script takes merged models as open_clip visual state dicts. A checkpoint is given as
the path of a ``merged.pt`` / ``model.pt`` written by ``if-merge``, as a baseline id (``wa``,
``ta``, ``iso_c``, ...; built like the initialization of ``if-merge clip``) or as
``pretrained``; models are named on the command line as ``NAME=CHECKPOINT``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from if_merge.clip.assets import ClipAssets, Head, normalize_images
from if_merge.clip.evaluate import head_logits, head_to
from if_merge.clip.model import State, build_exposed_visual, linear_layers, visual_state
from if_merge.clip.run import initial_state
from if_merge.clip.tasks import ARCHITECTURES

Models = dict[str, dict[str, torch.Tensor]]


def setting_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arch", required=True, choices=ARCHITECTURES)
    parser.add_argument("--tasks", required=True, type=int, choices=(8, 14, 20))
    parser.add_argument("--data", default=None, help="data root (default: $IF_MERGE_DATA or ./data)")
    parser.add_argument("--runs", default="runs", help="where baselines used as checkpoints are cached")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="also write the result to this JSON file")
    return parser


def load_state(checkpoint: str, args: argparse.Namespace, assets: ClipAssets) -> dict[str, torch.Tensor]:
    """``pretrained``, a baseline id or a checkpoint path -> float32 open_clip visual state dict."""
    if checkpoint == "pretrained":
        return {key: value.float() for key, value in visual_state(assets.load_checkpoint(None)).items()}
    return initial_state(checkpoint, args.arch, args.tasks, assets, args.device, Path(args.runs))[0]


def load_models(specs: list[str], args: argparse.Namespace, assets: ClipAssets) -> Models:
    """``NAME=CHECKPOINT`` arguments -> {name: state}; a bare checkpoint is named after itself."""
    models = {}
    for spec in specs:
        name, _, checkpoint = spec.rpartition("=")
        models[name or Path(checkpoint).stem] = load_state(checkpoint, args, assets)
    return models


def linear_weights(architecture: str, state: State) -> dict[str, torch.Tensor]:
    """The weight matrices IF-Merge fits, as [out, in] float32 tensors keyed by layer name.

    Query, key and value projections are separate matrices, the patch embedding is flattened
    and the output projection transposed: 6 matrices per Transformer block plus 2.
    """
    layers = linear_layers(build_exposed_visual(architecture, state))
    return {name: layer.weight.detach().float() for name, layer in layers.items()}


def test_subset(
    assets: ClipAssets, task: str, count: int, generator: torch.Generator, sort: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """The first ``count`` test images of ``task`` under a random permutation drawn from ``generator``."""
    images, labels = assets.load_images(task, "test")
    index = torch.randperm(len(labels), generator=generator)[:count]
    if sort:
        index = index.sort().values
    return images[index], labels[index]


@torch.no_grad()
def logits(
    visual: nn.Module,
    images: torch.Tensor,
    head: Head,
    device: str,
    half: bool = False,
    batch_size: int = 128,
) -> torch.Tensor:
    """Logits of a uint8 image cache; ``half`` runs the encoder under float16 autocast on CUDA."""
    chunks = []
    for start in range(0, len(images), batch_size):
        batch = normalize_images(images[start : start + batch_size], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=half and str(device).startswith("cuda")):
            features = visual(batch)
        chunks.append(head_logits(features, head))
    return torch.cat(chunks)


def shared_subsets(assets: ClipAssets, tasks, count: int) -> tuple[dict, dict]:
    """``count`` test images per task and the heads; one generator, seeded 0, visits the tasks in order."""
    generator = torch.Generator().manual_seed(0)
    subsets = {task: test_subset(assets, task, count, generator) for task in tasks}
    return subsets, {task: assets.load_head(task) for task in tasks}


@torch.no_grad()
def evaluate_subsets(architecture: str, state: State, subsets, heads, device: str, teachers=None) -> dict:
    """Per-task accuracy (%), cross-entropy and KL to the ``teachers`` logits, in float32."""
    visual = build_exposed_visual(architecture, state).to(device)
    result = {"accuracy": {}, "cross_entropy": {}, "expert_kl": {}, "logits": {}}
    for task, (images, labels) in subsets.items():
        scores = logits(visual, images, head_to(heads[task], device), device)
        labels = labels.to(device).long()
        result["logits"][task] = scores
        result["accuracy"][task] = float((scores.argmax(dim=-1) == labels).float().mean()) * 100
        result["cross_entropy"][task] = float(F.cross_entropy(scores, labels))
        if teachers is not None:
            teacher = teachers[task].log_softmax(dim=-1)
            kl = F.kl_div(scores.log_softmax(dim=-1), teacher, log_target=True, reduction="batchmean")
            result["expert_kl"][task] = float(kl)
    return result


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def interpolate(a: State, b: State, weight: float) -> dict[str, torch.Tensor]:
    """(1 - weight) a + weight b over every tensor of two state dicts."""
    return {key: (1.0 - weight) * a[key].float() + weight * b[key].float() for key in a}


def finish(result: dict, args: argparse.Namespace) -> None:
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))
        print(f"wrote {args.out}")
