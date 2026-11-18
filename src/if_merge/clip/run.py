"""Command line: IF-Merge on the CLIP benchmark, and evaluation of a checkpoint.

if-merge clip --arch ViT-B-32 --tasks 8 --init wa
if-merge clip --arch ViT-L-14 --tasks 20 --init iso_cts --seed 1
if-merge clip --arch ViT-B-32 --tasks 8 --init ta --set g.estimator=identity
if-merge evaluate --arch ViT-B-32 --tasks 8 --checkpoint runs/.../merged.pt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from ..config import IFMergeConfig
from ..core.merge import RoundRecord, if_merge
from .assets import ClipAssets, budget_sizes
from .backend import ClipBackend
from .evaluate import DiagnosticKL, evaluate
from .model import visual_state
from .tasks import ARCHITECTURES, task_names

RECIPE = Path(__file__).resolve().parents[3] / "configs" / "if_merge" / "clip.yaml"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arch", required=True, choices=ARCHITECTURES)
    parser.add_argument("--tasks", required=True, type=int, choices=(8, 14, 20))
    parser.add_argument("--data", default=None, help="data root (default: $IF_MERGE_DATA or ./data)")
    parser.add_argument("--runs", default="runs", help="directory for results and merged models")
    parser.add_argument("--device", default="cuda")


def setting_dir(runs: str | Path, architecture: str, task_count: int) -> Path:
    return Path(runs) / "clip" / f"{architecture}-{task_count}tasks"


def initial_state(name: str, architecture: str, task_count: int, assets: ClipAssets, device: str, runs: Path):
    """An open_clip visual state dict to start from: a baseline id or a checkpoint path."""
    if Path(name).is_file():
        state = torch.load(name, map_location="cpu", weights_only=True)
        return {k: v.float() for k, v in visual_state(state).items()}, {"checkpoint": name}
    if name == "wa":
        experts = [visual_state(assets.load_checkpoint(task)) for task in task_names(task_count)]
        mean = {key: torch.stack([expert[key].float() for expert in experts]).mean(0) for key in experts[0]}
        return mean, {"method": "wa"}
    from .baselines import build_initialization

    return build_initialization(name, architecture, task_count, assets, device, runs)


def run(args: argparse.Namespace) -> dict:
    overrides = list(args.set) + [f"calibration.seed={args.seed}"]
    if args.images is not None:
        samples, holdout = budget_sizes(args.images)
        overrides += [f"calibration.samples={samples}", f"calibration.holdout={holdout}"]
    config = IFMergeConfig.load(args.recipe, overrides)
    assets, tasks = ClipAssets(args.arch, args.data), task_names(args.tasks)
    splits = ("calib", "test", "val") if args.diagnostic_kl else ("calib", "test")
    missing = assets.missing(tasks, splits, seed=args.seed)
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} files, e.g. {missing[0]}. Run `if-merge prepare-clip` first."
        )

    name = args.name or "-".join(
        [re.sub(r"[^\w.+-]", "_", Path(args.init).stem), f"seed{args.seed}"]
        + ([f"images{args.images}"] if args.images not in (None, 512) else [])
        + [s.replace("=", "_") for s in args.set]
    )
    output = setting_dir(args.runs, args.arch, args.tasks) / "if_merge" / name
    if (output / "result.json").exists() and not args.force:
        print(f"[skip] {output / 'result.json'} exists (use --force to run again)")
        return json.loads((output / "result.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    started = time.monotonic()
    state, initialization = initial_state(
        args.init, args.arch, args.tasks, assets, args.device, Path(args.runs)
    )
    before = evaluate(args.arch, state, assets, tasks, args.device)
    print(f"[{name}] initialization {args.init}: {before['average']:.2f}", flush=True)

    diagnostic = DiagnosticKL(args.arch, assets, tasks, args.device) if args.diagnostic_kl else None
    diagnostic_kl = {"before": diagnostic(state)} if diagnostic else None

    def report(record: RoundRecord) -> None:
        print(f"[{name}] round {record.round}: holdout KL {record.holdout_kl:.5f}", flush=True)
        if args.save_rounds:
            torch.save(backend.state(), output / f"round{record.round}.pt")

    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    merge_started = time.monotonic()
    backend = ClipBackend(args.arch, tasks, assets, state, config, args.device)
    result = if_merge(backend, config, report)
    merge_seconds = time.monotonic() - merge_started
    backend = None  # release the merged model and the cached calibration data
    after = evaluate(args.arch, result.state, assets, tasks, args.device)
    if diagnostic:
        diagnostic_kl["after"] = diagnostic(result.state)
    print(
        f"[{name}] {before['average']:.2f} -> {after['average']:.2f} (round {result.selected_round})",
        flush=True,
    )

    record = {
        "architecture": args.arch,
        "tasks": list(tasks),
        "initialization": {"name": args.init, **initialization},
        "seed": args.seed,
        "recipe": config.to_dict(),
        "overrides": list(args.set),
        "before": before,
        "after": after,
        "gain": after["average"] - before["average"],
        "diagnostic_kl": diagnostic_kl,
        "rounds": [r.to_dict() for r in result.rounds],
        "selected_round": result.selected_round,
        "stop_reason": result.stop_reason,
        "merge_seconds": merge_seconds,
        "total_seconds": time.monotonic() - started,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated(args.device) / 2**30
            if str(args.device).startswith("cuda")
            else None
        ),
    }
    if not args.no_save:
        torch.save(result.state, output / "merged.pt")
    (output / "result.json").write_text(json.dumps(record, indent=2))
    return record


def run_evaluate(args: argparse.Namespace) -> dict:
    assets, tasks = ClipAssets(args.arch, args.data), task_names(args.tasks)
    state, _ = initial_state(args.checkpoint, args.arch, args.tasks, assets, args.device, Path(args.runs))
    scores = evaluate(args.arch, state, assets, tasks, args.device, split=args.split)
    for task, score in scores["per_task"].items():
        print(f"{task:15s} {score:6.2f}")
    print(f"{'average':15s} {scores['average']:6.2f}")
    return scores


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "clip", help="run IF-Merge on CLIP", description="Refine a merged CLIP visual encoder with IF-Merge."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--init",
        default="wa",
        help="initialization: a baseline id (wa, ta, ties, iso_cts, ...) or a checkpoint path",
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
        "--images",
        type=int,
        default=None,
        help="unlabeled images per task, |C_t| + |H_t| (default 512; sets calibration.samples and .holdout)",
    )
    parser.add_argument(
        "--diagnostic-kl",
        action="store_true",
        help="also report the expert-merged KL on validation images that take no part in the method",
    )
    parser.add_argument("--save-rounds", action="store_true", help="write the merged model after every round")
    parser.add_argument("--name", default=None, help="run name (default: <init>-seed<K>[-<overrides>])")
    parser.add_argument("--no-save", action="store_true", help="do not write merged.pt")
    parser.add_argument("--force", action="store_true", help="run even if the result exists")
    parser.set_defaults(run=run)

    parser = subparsers.add_parser("evaluate", help="evaluate a CLIP visual checkpoint or a baseline id")
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("test", "val"))
    parser.set_defaults(run=run_evaluate)
