"""Command line: IF-Merge, baselines and evaluation on the Flan-T5 benchmark.

    if-merge t5 --size base --init wa
    if-merge t5 --size base --init iso_cts --seed 1
    if-merge t5 --size base --init ta --set g.estimator=identity
    if-merge t5-baseline --size base --method tsv_m
    if-merge t5-baseline --size base --method ta --value 0.3
    if-merge t5-evaluate --size base --checkpoint runs/.../merged.safetensors

``docs/T5.md`` lists the command behind every row of the paper's Flan-T5 tables.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

from ..config import IFMergeConfig
from ..core.merge import RoundRecord, if_merge
from . import prepare
from .backend import T5Backend
from .baselines import METHODS, Workspace, build_initialization, run_baseline, selected_values, settings
from .data import SIZES, TASKS, T5Assets
from .evaluate import evaluate, evaluate_model
from .model import load_into, load_model, load_state, save_state

RECIPE = Path(__file__).resolve().parents[3] / "configs" / "if_merge" / "t5.yaml"
INITIALIZATIONS = ("wa", "ta", "iso_c", "iso_cts")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", default="base", choices=SIZES, help="Flan-T5-base or Flan-T5-large")
    parser.add_argument("--data", default=None, help="data root (default: $IF_MERGE_DATA or ./data)")
    parser.add_argument("--runs", default="runs", help="directory for results and merged models")
    parser.add_argument("--device", default="cuda")


def open_assets(args: argparse.Namespace) -> T5Assets:
    assets = T5Assets(args.size, args.data)
    missing = assets.missing()
    if missing:
        raise SystemExit(f"Missing {len(missing)} files, e.g. {missing[0]}. Run `if-merge prepare-t5` first.")
    return assets


def resources(device: str, started: float) -> dict:
    """Wall time and peak memory of this process (host memory on Linux only)."""
    cuda = str(device).startswith("cuda")
    try:
        import resource

        host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20 if sys.platform == "linux" else None
    except ImportError:
        host = None
    return {
        "total_seconds": time.monotonic() - started,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None,
        "peak_host_memory_gb": host,
    }


def environment() -> dict:
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__}


def initial_state(name: str, assets: T5Assets, device: str) -> tuple[dict[str, torch.Tensor], dict]:
    """A merged model to start from: a checkpoint path, or a baseline at the scale used in the paper."""
    if Path(name).is_file():
        return load_state(name), {"checkpoint": name}
    if name not in INITIALIZATIONS:
        raise SystemExit(
            f"--init must be one of {INITIALIZATIONS} or a .safetensors checkpoint; got {name!r}"
        )
    scale = selected_values(assets.size)["initializations"].get(name)  # weight averaging has no scale
    return build_initialization(name, Workspace.load(assets, device), scale), {"method": name, "scale": scale}


def run(args: argparse.Namespace) -> dict:
    overrides = list(args.set) + [f"calibration.seed={args.seed}"]
    config = IFMergeConfig.load(args.recipe, overrides)
    assets = open_assets(args)
    name = args.name or "-".join(
        [re.sub(r"[^\w.+-]", "_", Path(args.init).stem), f"seed{args.seed}"]
        + [override.replace("=", "_") for override in args.set]
        + (["freeze_head"] if args.freeze_head else [])
    )
    output = Path(args.runs) / "t5" / args.size / "if_merge" / name
    if (output / "result.json").exists() and not args.force:
        print(f"[skip] {output / 'result.json'} exists (use --force to run again)")
        return json.loads((output / "result.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    started = time.monotonic()
    state, initialization = initial_state(args.init, assets, args.device)
    before = evaluate(state, assets, args.device)
    print(f"[{name}] initialization {args.init}: {before['average']:.2f}", flush=True)

    def report(record: RoundRecord) -> None:
        print(
            f"[{name}] round {record.round}: step size {record.global_step}, "
            f"holdout KL {record.holdout_kl:.5f}",
            flush=True,
        )

    merge_started = time.monotonic()
    backend = T5Backend(assets, state, config, args.device, freeze_head=args.freeze_head)
    result = if_merge(backend, config, report)
    merge_seconds = time.monotonic() - merge_started
    del backend
    after = evaluate(result.state, assets, args.device)
    print(
        f"[{name}] {before['average']:.2f} -> {after['average']:.2f} (round {result.selected_round})",
        flush=True,
    )

    record = {
        "size": args.size,
        "tasks": list(TASKS),
        "initialization": {"name": args.init, **initialization},
        "seed": args.seed,
        "freeze_head": args.freeze_head,
        "recipe": config.to_dict(),
        "overrides": list(args.set),
        "before": before,
        "after": after,
        "gain": after["average"] - before["average"],
        "rounds": [r.to_dict() for r in result.rounds],
        "selected_round": result.selected_round,
        "stop_reason": result.stop_reason,
        "merge_seconds": merge_seconds,
        **resources(args.device, started),
        "environment": environment(),
    }
    if not args.no_save:
        save_state(result.state, output / "merged.safetensors")
    (output / "result.json").write_text(json.dumps(record, indent=2))
    return record


def run_baseline_command(args: argparse.Namespace) -> dict:
    assets = open_assets(args)
    name = f"seed{args.seed}" if args.value is None else f"value{args.value:g}-seed{args.seed}"
    output = Path(args.runs) / "t5" / args.size / "baselines" / args.method / name
    if (output / "result.json").exists() and not args.force:
        print(f"[skip] {output / 'result.json'} exists (use --force to run again)")
        return json.loads((output / "result.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    started = time.monotonic()
    workspace = Workspace.load(assets, args.device)
    state, details = run_baseline(args.method, workspace, args.seed, args.value)
    merge_seconds = time.monotonic() - started
    scores = evaluate_model(load_into(workspace.model, state), assets, args.device)
    print(f"[{args.method}] {scores['average']:.2f}", flush=True)

    record = {
        "size": args.size,
        "method": args.method,
        "seed": args.seed,
        "settings": {key: value for key, value in (settings()[args.method] or {}).items() if key != "grid"},
        **details,
        "validation": scores,
        "merge_seconds": merge_seconds,
        **resources(args.device, started),
        "environment": environment(),
    }
    if not args.no_save:
        save_state(state, output / "merged.safetensors")
    (output / "result.json").write_text(json.dumps(record, indent=2))
    return record


def run_evaluate(args: argparse.Namespace) -> dict:
    assets = open_assets(args)
    if args.checkpoint == "experts":  # every expert on its own task
        per_task = {}
        for task in TASKS:
            per_task.update(evaluate_model(load_model(assets, task), assets, args.device, [task])["per_task"])
        scores = {"average": sum(per_task.values()) / len(per_task), "per_task": per_task}
    elif args.checkpoint == "pretrained":
        scores = evaluate_model(load_model(assets), assets, args.device)
    else:
        scores = evaluate(load_state(args.checkpoint), assets, args.device)
    for task, score in scores["per_task"].items():
        print(f"{task:8s} {score:6.2f}")
    print(f"{'average':8s} {scores['average']:6.2f}")
    return scores


def register(subparsers) -> None:
    prepare.register(subparsers)

    parser = subparsers.add_parser(
        "t5", help="run IF-Merge on Flan-T5", description="Refine a merged Flan-T5 model with IF-Merge."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--init",
        default="wa",
        help=f"initialization: one of {', '.join(INITIALIZATIONS)} (at the scale used in the paper), "
        "or the path of a merged.safetensors",
    )
    parser.add_argument("--seed", type=int, default=0, help="calibration draw (C_t and H_t)")
    parser.add_argument("--recipe", default=str(RECIPE))
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="recipe override; repeatable",
    )
    parser.add_argument(
        "--freeze-head", action="store_true", help="keep the vocabulary head of the initialization"
    )
    parser.add_argument("--name", default=None, help="run name (default: <init>-seed<K>[-<overrides>])")
    parser.add_argument("--no-save", action="store_true", help="do not write merged.safetensors")
    parser.add_argument("--force", action="store_true", help="run even if the result exists")
    parser.set_defaults(run=run)

    parser = subparsers.add_parser("t5-baseline", help="run a merging baseline on Flan-T5")
    add_common_arguments(parser)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="draw of the calibration prompts and of the labeled selection split",
    )
    parser.add_argument(
        "--value", type=float, default=None, help="fix the hyperparameter instead of selecting it on the grid"
    )
    parser.add_argument("--no-save", action="store_true", help="do not write merged.safetensors")
    parser.add_argument("--force", action="store_true", help="run even if the result exists")
    parser.set_defaults(run=run_baseline_command)

    parser = subparsers.add_parser("t5-evaluate", help="evaluate a Flan-T5 checkpoint on GLUE validation")
    add_common_arguments(parser)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="a merged.safetensors, `pretrained`, or `experts` (each on its own task)",
    )
    parser.set_defaults(run=run_evaluate)
