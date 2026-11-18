"""Teacher-student distillation (KD) from the same initialization and images as IF-Merge.

    if-merge distill --arch ViT-B-32 --tasks 8 --init wa --images 512
    if-merge distill --arch ViT-B-16 --tasks 8 --init iso_cts --images 2048 --seed 1

The experts are the teachers. The merged model is trained with Adam on the mean over tasks
of KL(expert || merged model), each task under its own classification head, on the unlabeled
calibration images C_t. Only the weight matrices that IF-Merge fits are trained. The learning
rate and the number of steps are selected by the holdout KL on H_t, the criterion IF-Merge
uses for its step sizes and rounds; the test set is evaluated once, for the selected model.

Search protocol
    * every learning rate trains at a constant rate until its holdout KL has not improved for
      ``--patience`` steps, or for ``--max-steps`` steps;
    * while the best learning rate is the smallest or the largest one tried, the next value
      of the 1-3-10 grid beyond it is added (at most ``--extensions`` per side);
    * if the selected learning rate was still improving when it reached ``--max-steps``, it
      is trained again with twice as many steps (same batches).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .assets import ClipAssets, budget_sizes, calibration_split, normalize_images
from .evaluate import evaluate, head_logits, head_to
from .model import State, build_exposed_visual, linear_layers, open_clip_state
from .run import add_common_arguments, initial_state, setting_dir
from .tasks import task_names

DEFAULT_BUDGET = 512  # images per task of the default budget: |C_t| = |H_t| = 256
EVALUATION_BATCH = 128


def budget_split(images: int) -> tuple[list[int], list[int]]:
    """Positions of C_t and H_t inside a calibration draw, for a budget of ``images`` per task.

    Up to 512 images the budget is halved: the first half is C_t, the second half H_t (256 + 256
    is the default of IF-Merge). Larger budgets keep the holdout images of the default budget
    (positions 256-511) and add every further image to C_t.
    """
    return calibration_split(*budget_sizes(images))


def evaluation_interval(statistics_images: int, batch_size: int) -> int:
    """Steps between holdout evaluations: two passes over C_t, between 5 and 100 steps."""
    return min(100, max(5, 2 * math.ceil(statistics_images / batch_size)))


def next_learning_rate(learning_rate: float, direction: int) -> float:
    """The neighbor of ``learning_rate`` on the grid ..., 1e-6, 3e-6, 1e-5, 3e-5, ..."""
    exponent = math.floor(math.log10(learning_rate) + 1e-9)
    mantissa = learning_rate / 10**exponent
    if direction < 0:
        return (1.0 if mantissa > 2 else 3.0) * 10 ** (exponent if mantissa > 2 else exponent - 1)
    return (3.0 if mantissa < 2 else 1.0) * 10 ** (exponent if mantissa < 2 else exponent + 1)


def teacher_student_kl(teacher_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Mean over samples of KL(teacher || student)."""
    teacher = teacher_logits.log_softmax(dim=-1)
    student = logits.log_softmax(dim=-1)
    return (teacher.exp() * (teacher - student)).sum(dim=-1).mean()


@dataclass(frozen=True)
class SearchSettings:
    learning_rates: tuple[float, ...] = (3e-6, 1e-5, 3e-5)
    extensions: int = 2  # learning rates added per side while the best one is at the edge
    max_steps: int = 6000
    patience: int = 1500


@dataclass
class Trial:
    """One training run at a constant learning rate."""

    learning_rate: float
    best_kl: float
    best_step: int
    state: dict[str, torch.Tensor]  # the model at ``best_step`` (open_clip visual state dict)
    curve: list[tuple[int, float]]  # (step, holdout KL)
    stopped: str  # "patience", "max_steps" or "non_finite"
    last_step: int
    max_steps: int
    cap_limited: bool  # reached max_steps while the holdout KL was still improving
    seconds: float

    def summary(self) -> dict:
        return {name: value for name, value in vars(self).items() if name != "state"}


def search_learning_rates(
    train: Callable[[float, int], Trial], settings: SearchSettings
) -> tuple[dict[float, Trial], float]:
    """Run the search protocol; returns every trial and the selected learning rate."""

    def best(trials: dict[float, Trial]) -> float:
        return min(sorted(trials), key=lambda rate: (trials[rate].best_kl, rate))

    trials = {rate: train(rate, settings.max_steps) for rate in sorted(settings.learning_rates)}
    added = {-1: 0, 1: 0}
    while True:
        grid, selected = sorted(trials), best(trials)
        side = -1 if selected == grid[0] else (1 if selected == grid[-1] else 0)
        if side == 0 or added[side] >= settings.extensions:
            break
        added[side] += 1
        rate = next_learning_rate(selected, side)
        trials[rate] = train(rate, settings.max_steps)
    selected = best(trials)
    if trials[selected].cap_limited:
        trials[selected] = train(selected, 2 * settings.max_steps)
        selected = best(trials)
    return trials, selected


class Distillation:
    """Owns the calibration images, the experts' logits on them and the initialization."""

    def __init__(
        self,
        architecture: str,
        tasks: Sequence[str],
        assets: ClipAssets,
        initial_state: State,
        images: int = DEFAULT_BUDGET,
        seed: int = 0,
        device: str = "cuda",
        batch_size: int = 16,
        tasks_per_forward: int | None = None,
        patience: int = SearchSettings.patience,
    ) -> None:
        self.architecture = architecture
        self.tasks = tuple(tasks)
        self.initial_state = initial_state
        self.device = device
        self.batch_size = batch_size  # images per task per step
        self.patience = patience
        self.autocast = str(device).startswith("cuda")
        size = tasks_per_forward or len(self.tasks)
        self.groups = [self.tasks[start : start + size] for start in range(0, len(self.tasks), size)]
        self.heads = {task: head_to(assets.load_head(task), device) for task in self.tasks}

        statistics, holdout = budget_split(images)
        self.statistics_images, self.holdout_images = len(statistics), len(holdout)
        order = torch.tensor(statistics + holdout)  # C_t first, then H_t
        self.images, self.teacher_logits = {}, {}
        for task in self.tasks:
            draw, _ = assets.load_images(task, "calib", seed)
            if len(draw) < images:
                raise ValueError(
                    f"{task}: calibration draw {seed} has {len(draw)} images, {images} requested; "
                    f"build a longer one with `if-merge prepare-clip --calibration-images {images}`"
                )
            self.images[task] = draw[order].contiguous().to(device)
            expert = build_exposed_visual(architecture, assets.load_checkpoint(task)).to(device)
            self.teacher_logits[task] = torch.cat(
                [
                    self._evaluation_logits(expert, task, self.images[task][start : start + EVALUATION_BATCH])
                    for start in range(0, images, EVALUATION_BATCH)
                ]
            )
            del expert
        if self.autocast:
            torch.cuda.empty_cache()

    def _features(self, visual, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.autocast):
            return visual(normalize_images(images, self.device))

    @torch.no_grad()
    def _evaluation_logits(self, visual, task: str, images: torch.Tensor) -> torch.Tensor:
        """Logits computed entirely in float16 on CUDA, classification head included."""
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.autocast):
            return head_logits(visual(normalize_images(images, self.device)), self.heads[task]).float()

    @torch.no_grad()
    def holdout_kl(self, visual) -> float:
        """Teacher-student KL on H_t, averaged over tasks."""
        total = 0.0
        for task in self.tasks:
            images = self.images[task][self.statistics_images :]
            teacher = self.teacher_logits[task][self.statistics_images :]
            kl = 0.0
            for start in range(0, len(images), EVALUATION_BATCH):
                batch = slice(start, start + EVALUATION_BATCH)
                logits = self._evaluation_logits(visual, task, images[batch])
                kl += float(teacher_student_kl(teacher[batch], logits)) * len(logits)
            total += kl / len(images)
        return total / len(self.tasks)

    def train(self, learning_rate: float, max_steps: int, report: Callable[[str], None] = print) -> Trial:
        """Train from the initialization at a constant learning rate; keep the best holdout KL."""
        visual = build_exposed_visual(self.architecture, self.initial_state).to(self.device)
        for parameter in visual.parameters():
            parameter.requires_grad_(False)
        weights = [layer.weight for layer in linear_layers(visual).values()]
        for weight in weights:
            weight.requires_grad_(True)
        optimizer = torch.optim.Adam(weights, lr=learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=self.autocast)
        generator = torch.Generator().manual_seed(0)  # every learning rate sees the same batches
        interval = evaluation_interval(self.statistics_images, self.batch_size)

        best_kl, best_step, best_state = self.holdout_kl(visual), 0, open_clip_state(visual)
        curve = [(0, best_kl)]
        step, stopped, started = 0, "max_steps", time.monotonic()
        while step < max_steps:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            for group in self.groups:
                index = {
                    task: torch.randint(0, self.statistics_images, (self.batch_size,), generator=generator)
                    for task in group
                }
                batch = torch.cat([self.images[task][index[task].to(self.device)] for task in group])
                features = self._features(visual, batch)
                loss = 0.0
                for position, task in enumerate(group):
                    rows = slice(position * self.batch_size, (position + 1) * self.batch_size)
                    logits = head_logits(features[rows], self.heads[task])
                    teacher = self.teacher_logits[task][index[task].to(self.device)]
                    loss = loss + teacher_student_kl(teacher, logits)
                scaler.scale(loss / len(self.tasks)).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % interval:
                continue
            kl = self.holdout_kl(visual)
            curve.append((step, kl))
            if kl < best_kl:
                best_kl, best_step, best_state = kl, step, open_clip_state(visual)
            report(
                f"lr {learning_rate:.0e} step {step}: holdout KL {kl:.5f} (best {best_kl:.5f} at {best_step})"
            )
            if not math.isfinite(kl):
                stopped = "non_finite"
                break
            if step - best_step >= self.patience:
                stopped = "patience"
                break
        del visual, optimizer
        if self.autocast:
            torch.cuda.empty_cache()
        return Trial(
            learning_rate=learning_rate,
            best_kl=best_kl,
            best_step=best_step,
            state=best_state,
            curve=curve,
            stopped=stopped,
            last_step=step,
            max_steps=max_steps,
            cap_limited=stopped == "max_steps" and step - best_step < self.patience,
            seconds=time.monotonic() - started,
        )


def run(args: argparse.Namespace) -> dict:
    assets, tasks = ClipAssets(args.arch, args.data), task_names(args.tasks)
    missing = assets.missing(tasks, seed=args.seed)
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} files, e.g. {missing[0]}. Run `if-merge prepare-clip` first."
        )
    stem = re.sub(r"[^\w.+-]", "_", Path(args.init).stem)
    name = args.name or f"{stem}-seed{args.seed}-n{args.images}"
    output = setting_dir(args.runs, args.arch, args.tasks) / "distill" / name
    if (output / "result.json").exists() and not args.force:
        print(f"[skip] {output / 'result.json'} exists (use --force to run again)")
        return json.loads((output / "result.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)  # --seed only selects the calibration draw
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.monotonic()
    state, initialization = initial_state(
        args.init, args.arch, args.tasks, assets, args.device, Path(args.runs)
    )
    before = evaluate(args.arch, state, assets, tasks, args.device)
    print(f"[{name}] initialization {args.init}: {before['average']:.2f}", flush=True)

    settings = SearchSettings(tuple(args.learning_rates), args.extensions, args.max_steps, args.patience)
    distillation = Distillation(
        args.arch,
        tasks,
        assets,
        state,
        args.images,
        args.seed,
        args.device,
        args.batch_size,
        args.tasks_per_forward,
        args.patience,
    )

    def train(learning_rate: float, max_steps: int) -> Trial:
        return distillation.train(
            learning_rate, max_steps, lambda line: print(f"[{name}] {line}", flush=True)
        )

    trials, selected = search_learning_rates(train, settings)
    chosen = trials[selected]
    after = evaluate(args.arch, chosen.state, assets, tasks, args.device)
    print(
        f"[{name}] {before['average']:.2f} -> {after['average']:.2f} "
        f"(learning rate {selected:.0e}, step {chosen.best_step})",
        flush=True,
    )

    record = {
        "architecture": args.arch,
        "tasks": list(tasks),
        "initialization": {"name": args.init, **initialization},
        "seed": args.seed,
        "images_per_task": args.images,
        "statistics_images": distillation.statistics_images,
        "holdout_images": distillation.holdout_images,
        "before": before,
        "after": after,
        "gain": after["average"] - before["average"],
        "selected_learning_rate": selected,
        "selected_step": chosen.best_step,
        "selected_holdout_kl": chosen.best_kl,
        "selected_at_grid_edge": selected in (min(trials), max(trials)),
        "selected_cap_limited": chosen.cap_limited,
        "trials": {f"{rate:.0e}": trials[rate].summary() for rate in sorted(trials)},
        "search": {
            "learning_rates": list(settings.learning_rates),
            "extensions": settings.extensions,
            "max_steps": settings.max_steps,
            "patience": settings.patience,
            "batch_size_per_task": args.batch_size,
            "evaluation_interval": evaluation_interval(distillation.statistics_images, args.batch_size),
        },
        "total_seconds": time.monotonic() - started,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated(args.device) / 2**30
            if str(args.device).startswith("cuda")
            else None
        ),
    }
    if not args.no_save:
        torch.save(chosen.state, output / "merged.pt")
    (output / "result.json").write_text(json.dumps(record, indent=2))
    return record


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "distill",
        help="teacher-student distillation from the same initialization and images as IF-Merge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--init",
        default="wa",
        help="initialization: a baseline id (wa, ta, iso_cts, ...) or a checkpoint path",
    )
    parser.add_argument("--seed", type=int, default=0, help="calibration draw (C_t and H_t)")
    parser.add_argument(
        "--images",
        type=int,
        default=DEFAULT_BUDGET,
        help="unlabeled images per task, |C_t| + |H_t| (default 512)",
    )
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=list(SearchSettings.learning_rates), metavar="LR"
    )
    parser.add_argument("--extensions", type=int, default=SearchSettings.extensions)
    parser.add_argument("--max-steps", type=int, default=SearchSettings.max_steps)
    parser.add_argument("--patience", type=int, default=SearchSettings.patience)
    parser.add_argument("--batch-size", type=int, default=16, help="images per task per step")
    parser.add_argument(
        "--tasks-per-forward",
        type=int,
        default=None,
        help="split a step into several forward passes (memory)",
    )
    parser.add_argument("--name", default=None, help="run name (default: <init>-seed<K>-n<N>)")
    parser.add_argument("--no-save", action="store_true", help="do not write merged.pt")
    parser.add_argument("--force", action="store_true", help="run even if the result exists")
    parser.set_defaults(run=run)
