"""Download everything the Flan-T5 benchmark needs.

    if-merge prepare-t5 --size base
    if-merge prepare-t5 --size large

Sources (Hugging Face)
    pretrained models   google/flan-t5-<size>
    base experts        tanganke/flan-t5-base_glue-<task>            fully fine-tuned
    large experts       tanganke/flan-t5-large_glue-<task>_lora-16   LoRA adapters (rank 16)
    data                nyu-mll/glue

The large experts are stored as full weights: the adapters are merged into the
pretrained model once, W + (alpha / r) B A, so that both sizes are handled alike.
Tokenized splits are built on first use (see ``if_merge.t5.data``).
"""

from __future__ import annotations

import argparse
import json

import torch

from .data import SIZES, TASKS, T5Assets
from .model import load_model

MODEL_FILES = ["*.safetensors", "*.json", "spiece.model"]


@torch.no_grad()
def merge_adapter(model: torch.nn.Module, adapter_directory) -> torch.nn.Module:
    """Add the low-rank updates of a LoRA adapter to the weights of ``model``, in place."""
    from safetensors.torch import load_file

    config = json.loads((adapter_directory / "adapter_config.json").read_text())
    scaling = config["lora_alpha"] / config["r"]
    adapter = load_file(str(adapter_directory / "adapter_model.safetensors"))
    for key, down in adapter.items():
        if key.endswith("lora_A.weight"):
            layer = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
            up = adapter[key.replace("lora_A", "lora_B")]
            model.get_submodule(layer).weight += (up @ down) * scaling
    return model


def prepare(size: str, root=None) -> None:
    from huggingface_hub import snapshot_download

    assets = T5Assets(size, root)
    snapshot_download(f"google/flan-t5-{size}", local_dir=assets.model_dir(), allow_patterns=MODEL_FILES)
    print(f"pretrained  {assets.model_dir()}", flush=True)
    for task in TASKS:
        if (assets.model_dir(task) / "config.json").exists():
            continue
        if size == "base":
            repository = f"tanganke/flan-t5-base_glue-{task}"
            snapshot_download(repository, local_dir=assets.model_dir(task), allow_patterns=MODEL_FILES)
        else:
            repository = f"tanganke/flan-t5-{size}_glue-{task}_lora-16"
            snapshot_download(repository, local_dir=assets.adapter_dir(task), allow_patterns=MODEL_FILES)
            expert = merge_adapter(load_model(assets), assets.adapter_dir(task))
            expert.save_pretrained(assets.model_dir(task), safe_serialization=True)
        print(f"expert      {assets.model_dir(task)}", flush=True)
    snapshot_download(
        "nyu-mll/glue",
        repo_type="dataset",
        local_dir=assets.t5 / "glue",
        allow_patterns=[f"{task}/*.parquet" for task in TASKS],
    )
    print(f"data        {assets.t5 / 'glue'}", flush=True)
    missing = assets.missing()
    if missing:
        raise SystemExit(f"Still missing after the download: {missing}")


def run(args: argparse.Namespace) -> None:
    for size in args.size:
        prepare(size, args.data)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "prepare-t5", help="download Flan-T5, the GLUE experts and GLUE", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )  # fmt: skip
    parser.add_argument("--size", nargs="+", default=["base"], choices=SIZES)
    parser.add_argument("--data", default=None, help="data root (default: $IF_MERGE_DATA or ./data)")
    parser.set_defaults(run=run)
