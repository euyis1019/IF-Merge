"""The merging baselines of the paper on the CLIP benchmark.

    if-merge baseline --arch ViT-B-32 --tasks 8 --method tsv_m
    if-merge baseline --arch ViT-B-32 --tasks 8 --method reforge --init iso_cts
    if-merge baseline --list

``run_baseline`` follows the paper's protocol: the hyperparameters of a method are
selected on the labeled validation split over the grid in ``configs/baselines/clip.yaml``
and the selected merged model is evaluated once on the test split.
``build_initialization`` returns a baseline (or a checkpoint) as the initialization of
IF-Merge, using the selections stored in ``configs/baselines/clip_selected.yaml``.

Two layouts of the visual encoder are used. Data-free methods and the post-hoc LiNeS and
SVC work on the open_clip state dict, where the query/key/value projections of a block
are one packed matrix. Methods that read layer inputs (RegMean, RegMean++, Fisher
Merging, MaTS, AdaMerging, ReForge) work on the encoder with split attention
projections. Each merged model is evaluated in the layout it was built in; under float16
autocast the two layouts can disagree on a few test images.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from itertools import product
from pathlib import Path

import torch
import yaml
from torch.func import functional_call

from ..baselines import BASELINES
from ..baselines.adamerging import adamerging
from ..baselines.fisher import diagonal_fisher, fisher_merge, mats
from ..baselines.posthoc import SingularValueCalibration, lines
from ..baselines.reforge import GaussianProcessOptimizer, anchored_regression, depth_groups
from ..baselines.regmean import regmean, regmean_plus_plus
from ..baselines.spectral import iso_c, iso_cts, tsv_m, wudi
from ..baselines.weight_space import task_arithmetic, ties, weight_average
from ..core.statistics import StatisticsCollector
from .assets import ClipAssets, Head, normalize_images
from .evaluate import accuracy, classify, head_logits, head_to
from .model import build_exposed_visual, build_visual, linear_layers, open_clip_state, visual_state
from .tasks import task_names

logger = logging.getLogger(__name__)

CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "baselines"

State = dict[str, torch.Tensor]
Init = str | Path | None
#: (merged state in the layout of the method, selected hyperparameters, validation sweep)
Outcome = tuple[State, dict, list[dict]]
Runner = Callable[["_Benchmark", dict, Init], Outcome]

#: methods that merge task vectors without data; they also define the merged task vector
#: a post-hoc method starts from
TASK_VECTOR_METHODS: dict[str, Callable] = {
    "wa": weight_average,
    "ta": task_arithmetic,
    "ties": ties,
    "tsv_m": tsv_m,
    "wudi": wudi,
    "iso_c": iso_c,
    "iso_cts": iso_cts,
}
_ON_DEVICE = ("tsv_m", "wudi", "iso_c", "iso_cts")


# ------------------------------------------------------------------ settings
def _load_config(name: str) -> dict:
    with (CONFIGS / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def baseline_settings(method: str, architecture: str, task_count: int, overrides: dict | None = None) -> dict:
    """Grid and fixed hyperparameters of ``method`` in one setting, with ``overrides`` applied."""
    settings = dict(_load_config("clip.yaml")[method] or {})
    settings.update(overrides or {})
    return settings


def selected_hyperparameters(
    method: str, architecture: str, task_count: int, init: str | None = None
) -> dict | None:
    """The hyperparameters selected on the validation split for this setting, if stored."""
    key = method if init is None else f"{method}+{init}"
    return _load_config("clip_selected.yaml").get(key, {}).get(architecture, {}).get(task_count)


def baseline_directory(
    runs: str | Path, architecture: str, task_count: int, method: str, init: Init = None
) -> Path:
    """``<runs>/clip/<arch>-<T>tasks/baselines/<method>[+<init>]``."""
    name = method if init is None else f"{method}+{init if str(init) in BASELINES else Path(init).stem}"
    return Path(runs) / "clip" / f"{architecture}-{task_count}tasks" / "baselines" / name


# ----------------------------------------------------------------- benchmark
class _Benchmark:
    """Experts, heads, images and evaluation of one benchmark setting."""

    def __init__(self, architecture: str, task_count: int, assets: ClipAssets, device: str, seed: int = 0):
        self.architecture, self.assets, self.device, self.seed = architecture, assets, device, seed
        self.tasks = task_names(task_count)
        self.pretrained = _float_visual(assets.load_checkpoint(None))
        self.experts = [_float_visual(assets.load_checkpoint(task)) for task in self.tasks]
        self.heads: dict[str, Head] = {task: assets.load_head(task) for task in self.tasks}
        self._modules: dict[bool, torch.nn.Module] = {}
        self._images: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._split_states: tuple[State, list[State]] | None = None

    # ---- the two layouts
    def split(self, state: State) -> State:
        """A state dict in the layout with split attention projections."""
        visual = build_exposed_visual(self.architecture, state, boundary=False)
        return {name: tensor.clone() for name, tensor in visual.state_dict().items()}

    def _split_all(self) -> tuple[State, list[State]]:
        if self._split_states is None:
            self._split_states = (
                self.split(self.pretrained),
                [self.split(expert) for expert in self.experts],
            )
        return self._split_states

    @property
    def split_pretrained(self) -> State:
        return self._split_all()[0]

    @property
    def split_experts(self) -> list[State]:
        return self._split_all()[1]

    def module(self, split: bool) -> torch.nn.Module:
        """The encoder used for evaluation in either layout."""
        if split not in self._modules:
            self._modules[split] = (
                build_exposed_visual(self.architecture, self.pretrained, boundary=False)
                if split
                else build_visual(self.architecture, self.pretrained)
            )
        return self._modules[split]

    @property
    def layer_names(self) -> list[str]:
        """The weight matrices fitted by the activation-based methods (the block linears)."""
        return list(linear_layers(self.module(split=True)))

    def task_vectors(self, split: bool = False) -> list[State]:
        pretrained = self.split_pretrained if split else self.pretrained
        experts = self.split_experts if split else self.experts
        return [
            {name: expert[name].float() - base.float() for name, base in pretrained.items()}
            for expert in experts
        ]

    # ---- data
    def calibration(self, task: str, samples: int) -> torch.Tensor:
        """The first ``samples`` unlabeled calibration images of ``task`` (uint8)."""
        if (task, "calib") not in self._images:
            self._images[task, "calib"] = self.assets.load_images(task, "calib", self.seed)
        images = self._images[task, "calib"][0]
        if len(images) < samples:
            raise ValueError(f"{task}: {len(images)} calibration images, {samples} requested")
        return images[:samples]

    def evaluate(self, state: State, split: str) -> dict:
        """Per-task and task-averaged accuracy (%) of ``state``, in the layout of its keys."""
        visual = self.module(split=any(".attn.q_proj." in name for name in state))
        visual.load_state_dict(state, strict=True)
        per_task = {}
        for task in self.tasks:
            if split == "val" and (task, split) not in self._images:  # reused by every grid point
                self._images[task, split] = self.assets.load_images(task, split)
            images, labels = self._images.get((task, split)) or self.assets.load_images(task, split)
            per_task[task] = accuracy(visual, self.heads[task], images, labels, self.device)
        return {"average": sum(per_task.values()) / len(per_task), "per_task": per_task}


def _float_visual(state: State) -> State:
    """The visual encoder of a CLIP or visual state dict, in float32."""
    return {name: tensor.float() for name, tensor in visual_state(state).items()}


def _load_state(path: str | Path) -> State:
    return _float_visual(torch.load(path, map_location="cpu", weights_only=True))


def _empty_cache(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


# ----------------------------------------------------------------- selection
def _grid(settings: dict) -> list[dict]:
    grid = settings.get("grid", {})
    return [dict(zip(grid, values, strict=True)) for values in product(*grid.values())]


def _select(bench: _Benchmark, settings: dict, build: Callable[..., State]) -> Outcome:
    """Build the merged model of every grid point and keep the best on the validation split.

    Ties go to the earlier grid point. ``settings["selected"]`` skips the selection.
    """
    if "selected" in settings:
        return build(**settings["selected"]), dict(settings["selected"]), []
    sweep, best = [], None
    for hyperparameters in _grid(settings):
        try:
            state = build(**hyperparameters)
        except torch.linalg.LinAlgError:  # a singular system: the grid point is rejected
            sweep.append({**hyperparameters, "val_average": None})
            continue
        average = bench.evaluate(state, "val")["average"]
        sweep.append({**hyperparameters, "val_average": average})
        logger.info("%s: validation %.2f", hyperparameters, average)
        if best is None or average > best[0]:
            best = (average, hyperparameters, state)
    if best is None:
        raise RuntimeError("No grid point produced a merged model.")
    return best[2], best[1], sweep


# ---------------------------------------------------------- data-free methods
def _merged_task_vector(bench: _Benchmark, method: str, task_vectors: list[State], settings: dict) -> State:
    fixed = {name: value for name, value in settings.items() if name not in ("grid", "selected")}
    if method in _ON_DEVICE:
        fixed["device"] = bench.device
    return TASK_VECTOR_METHODS[method](task_vectors, **fixed)


def _scaled(pretrained: State, task_vector: State) -> Callable[..., State]:
    def build(scale: float) -> State:
        return {name: base + scale * task_vector[name] for name, base in pretrained.items()}

    return build


def _run_weight_average(bench: _Benchmark, settings: dict, init: Init) -> Outcome:
    names = bench.pretrained
    return (
        {name: torch.stack([expert[name] for expert in bench.experts]).mean(dim=0) for name in names},
        {},
        [],
    )


def _task_vector_runner(method: str) -> Runner:
    def run(bench: _Benchmark, settings: dict, init: Init) -> Outcome:
        merged = _merged_task_vector(bench, method, bench.task_vectors(), settings)
        return _select(bench, settings, _scaled(bench.pretrained, merged))

    return run


# ------------------------------------------------------------ post-hoc methods
def _initial_task_vector(bench: _Benchmark, init: Init, split: bool) -> State:
    """The merged task vector a post-hoc method starts from, before any scaling.

    For a data-free baseline this is its merged task vector, computed in the requested
    layout; for a checkpoint or another baseline, the difference to the pretrained model.
    """
    if str(init) in TASK_VECTOR_METHODS:
        settings = baseline_settings(str(init), bench.architecture, len(bench.tasks))
        return _merged_task_vector(bench, str(init), bench.task_vectors(split), settings)
    state, _ = build_initialization(
        str(init), bench.architecture, len(bench.tasks), bench.assets, bench.device
    )
    pretrained = bench.split_pretrained if split else bench.pretrained
    state = bench.split(state) if split else state
    return {name: state[name].float() - base.float() for name, base in pretrained.items()}


def _block_index(name: str) -> int | None:
    match = re.search(r"resblocks\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _run_lines(bench: _Benchmark, settings: dict, init: Init) -> Outcome:
    scaled = lines(_initial_task_vector(bench, init, split=False), _block_index, settings["depth_ratio"])
    return _select(bench, settings, _scaled(bench.pretrained, scaled))


def _run_svc(bench: _Benchmark, settings: dict, init: Init) -> Outcome:
    merged = _initial_task_vector(bench, init, split=False)
    calibration = SingularValueCalibration(bench.task_vectors(), merged, bench.device)

    def build(strength: float, scale: float) -> State:
        calibrated = calibration.apply(scale, strength)
        return {name: base + calibrated[name] for name, base in bench.pretrained.items()}

    return _select(bench, settings, build)


def _run_reforge(bench: _Benchmark, settings: dict, init: Init) -> Outcome:
    pretrained, experts, layers = bench.split_pretrained, bench.split_experts, bench.layer_names
    anchor_vector = _initial_task_vector(bench, init, split=True)
    grams = _expert_grams(bench, settings["samples"])
    groups, bounds = settings["groups"], settings["bounds"]
    group_of = depth_groups(layers, groups)
    per_group = ("log_ridge", "off_diagonal_scale", "output_scale")
    weights = [f"{layer}.weight" for layer in layers]

    def build(point: list[float]) -> State:
        anchor = {name: pretrained[name].float() + point[-1] * anchor_vector[name] for name in weights}
        columns = [point[offset : 3 * groups : 3] for offset in range(3)]
        return anchored_regression(
            pretrained, experts, grams, layers, anchor, *columns, group_of, bench.device
        )

    def named(point: list[float]) -> dict:
        selected = {name: point[offset : 3 * groups : 3] for offset, name in enumerate(per_group)}
        return {**selected, "anchor_scale": point[-1]}

    if "selected" in settings:
        selected = settings["selected"]
        point = [selected[name][group] for group in range(groups) for name in per_group]
        return build([*point, selected["anchor_scale"]]), dict(selected), []

    search_space = [tuple(bounds[name]) for _ in range(groups) for name in per_group]
    search_space.append(tuple(bounds["anchor_scale"]))
    optimizer = GaussianProcessOptimizer(search_space, seed=bench.seed)
    sweep = []
    for evaluation in range(settings["evaluations"]):
        point = optimizer.ask()
        average = bench.evaluate(build(point), "val")["average"]
        optimizer.tell(point, average)
        sweep.append({**named(point), "val_average": average})
        logger.info(
            "evaluation %d: validation %.2f (best %.2f)", evaluation + 1, average, optimizer.best()[1]
        )
    point, _ = optimizer.best()
    return build(point), named(point), sweep


# ------------------------------------------------- methods with calibration data
@torch.no_grad()
def _expert_grams(bench: _Benchmark, samples: int, batch_size: int = 64) -> list[dict[str, torch.Tensor]]:
    """Gram matrices of the linear-layer inputs of every expert on its own calibration images."""
    grams = []
    for task, expert in zip(bench.tasks, bench.experts, strict=True):
        visual = build_exposed_visual(bench.architecture, expert, boundary=False).to(bench.device)
        collector = StatisticsCollector(linear_layers(visual), feedback=False)
        images = bench.calibration(task, samples)
        with collector:
            for start in range(0, len(images), batch_size):
                visual(normalize_images(images[start : start + batch_size], bench.device))
        grams.append({name: moment.gram.cpu() for name, moment in collector.inputs.items()})
        del visual, collector
        _empty_cache(bench.device)
    return grams


def _run_regmean(bench: _Benchmark, settings: dict, init: Init) -> Outcome:
    experts, layers = bench.split_experts, bench.layer_names
    grams = _expert_grams(bench, settings["samples"])

    def build(off_diagonal_scale: float) -> State:
        return regmean(experts, grams, layers, off_diagonal_scale, bench.device)

    return _select(bench, settings, build)


def _run_regmean_plus_plus(bench: _Benchmark, settings: dict, init: Init) -> Outcome:
    device, task_count, batch_size = bench.device, len(bench.tasks), 32
    expert_blocks, expert_rest = [], []
    for expert in bench.split_experts:
        blocks: dict[int, State] = {}
        for name, tensor in expert.items():
            if (index := _block_index(name)) is not None:
                blocks.setdefault(index, {})[name.split(f"resblocks.{index}.", 1)[1]] = tensor
        expert_blocks.append([blocks[index] for index in sorted(blocks)])
        expert_rest.append({name: tensor for name, tensor in expert.items() if _block_index(name) is None})
    images = [bench.calibration(task, settings["samples"]) for task in bench.tasks]
    scratch = build_exposed_visual(bench.architecture, bench.pretrained, boundary=False).to(device)

    @torch.no_grad()
    def build(off_diagonal_scale: float) -> State:
        merged = build_exposed_visual(bench.architecture, bench.pretrained, boundary=False).to(device)
        state = merged.state_dict()
        for name in expert_rest[0]:  # parameters outside the blocks: expert average
            state[name].copy_(sum(rest[name].float() for rest in expert_rest).div(task_count))
        inputs = [
            torch.cat(
                [
                    merged._embeds(normalize_images(task_images[start : start + batch_size], device)).float()
                    for start in range(0, len(task_images), batch_size)
                ]
            )
            for task_images in images
        ]
        regmean_plus_plus(
            merged.transformer.resblocks,
            expert_blocks,
            inputs,
            scratch.transformer.resblocks[0],
            off_diagonal_scale,
            batch_size,
        )
        return {name: tensor.detach().float().cpu() for name, tensor in merged.state_dict().items()}

    return _select(bench, settings, build)


def _logit_batches(
    visual, images: torch.Tensor, head: Head, batch_size: int, device: str
) -> Iterator[torch.Tensor]:
    for start in range(0, len(images), batch_size):
        yield classify(visual, normalize_images(images[start : start + batch_size], device), head)


def _run_fisher(bench: _Benchmark, settings: dict, init: Init) -> Outcome:
    fishers = []
    for task, expert in zip(bench.tasks, bench.experts, strict=True):
        visual = build_exposed_visual(bench.architecture, expert, boundary=False).to(bench.device)
        visual.requires_grad_(True)
        images = bench.calibration(task, settings["samples"])
        head = head_to(bench.heads[task], bench.device)
        batches = _logit_batches(visual, images, head, settings["batch_size"], bench.device)
        fishers.append(diagonal_fisher(visual, batches))
        del visual
        _empty_cache(bench.device)
    return fisher_merge(bench.split_experts, fishers, settings["floor"], settings["normalize"]), {}, []


def _run_mats(bench: _Benchmark, settings: dict, init: Init) -> Outcome:
    """MaTS started from the Task Arithmetic model with the validation-selected scale."""
    if "selected" in settings:
        scale, sweep = settings["selected"]["ta_scale"], []
    else:
        ta_settings = baseline_settings("ta", bench.architecture, len(bench.tasks))
        _, selected, sweep = _task_vector_runner("ta")(bench, ta_settings, None)
        scale = selected["scale"]
        sweep = [{"ta_scale": entry["scale"], "val_average": entry["val_average"]} for entry in sweep]
    pretrained, experts = bench.split_pretrained, bench.split_experts
    grams = _expert_grams(bench, settings["samples"])
    initial = {
        name: base.float() + scale * sum(expert[name].float() - base.float() for expert in experts)
        for name, base in pretrained.items()
    }
    state = mats(experts, grams, bench.layer_names, initial, settings["iterations"], bench.device)
    return state, {"ta_scale": scale}, sweep


def _log_entropy(step: int, entropy: float) -> None:
    if step % 50 == 0:
        logger.info("step %d: mean entropy %.4f", step, entropy)


def _run_adamerging(bench: _Benchmark, settings: dict, init: Init) -> Outcome:
    device, batch_size = bench.device, settings["batch_size"]
    torch.manual_seed(bench.seed)
    # The encoder is frozen here and stays frozen when the merged model is evaluated on it.
    # Float16 autocast caches the casts of trainable weights only, and the two code paths
    # differ in a few predictions; the reference numbers were computed on the frozen encoder.
    visual = bench.module(split=True).to(device)
    visual.requires_grad_(False)
    pools = [bench.calibration(task, settings["samples"]) for task in bench.tasks]
    heads = [head_to(bench.heads[task], device) for task in bench.tasks]
    generator = torch.Generator(device="cpu").manual_seed(bench.seed)

    def task_logits(task: int, state: State) -> torch.Tensor:
        index = torch.randint(0, len(pools[task]), (batch_size,), generator=generator)
        images = normalize_images(pools[task][index], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            features = functional_call(visual, state, (images,))
        return head_logits(features, heads[task])

    state = adamerging(
        bench.split_pretrained,
        bench.task_vectors(split=True),
        task_logits,
        settings["steps"],
        settings["learning_rate"],
        settings["initial_coefficient"],
        device,
        _log_entropy,
    )
    return {name: tensor.float().cpu() for name, tensor in state.items()}, {}, []


_RUNNERS: dict[str, Runner] = {
    "wa": _run_weight_average,
    **{method: _task_vector_runner(method) for method in TASK_VECTOR_METHODS if method != "wa"},
    "regmean": _run_regmean,
    "regmeanpp": _run_regmean_plus_plus,
    "fisher": _run_fisher,
    "mats": _run_mats,
    "adamerging": _run_adamerging,
    "lines": _run_lines,
    "svc": _run_svc,
    "reforge": _run_reforge,
}


# ---------------------------------------------------------------- public API
def run_baseline(
    method: str,
    architecture: str,
    task_count: int,
    assets: ClipAssets,
    device: str,
    output_dir: Path | None = None,
    init: str | Path | None = None,
    seed: int = 0,
    overrides: dict | None = None,
) -> dict:
    """Run one baseline with the paper's protocol and return its result record.

    ``init`` is the merged model a post-hoc method (LiNeS, SVC, ReForge) starts from: a
    baseline id or the path of a checkpoint. ``seed`` selects the draw of unlabeled
    calibration images and seeds the stochastic methods (AdaMerging, ReForge).
    ``overrides`` replaces entries of the method's settings in ``configs/baselines/clip.yaml``;
    ``{"selected": {...}}`` fixes the hyperparameters and skips the validation sweep.
    With ``output_dir``, writes ``result.json`` and ``merged.pt`` (open_clip visual state
    dict, float32).
    """
    if method not in BASELINES:
        raise ValueError(f"Unknown baseline {method!r}; expected one of {list(BASELINES)}")
    if BASELINES[method].posthoc != (init is not None):
        expected = "needs" if BASELINES[method].posthoc else "does not take"
        raise ValueError(f"{method} {expected} an initialization (init).")
    settings = baseline_settings(method, architecture, task_count, overrides)

    started = time.monotonic()
    on_gpu = str(device).startswith("cuda")
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(device)
    bench = _Benchmark(architecture, task_count, assets, device, seed)
    state, selected, sweep = _RUNNERS[method](bench, settings, init)
    test = bench.evaluate(state, "test")

    scores = [entry["val_average"] for entry in sweep if entry["val_average"] is not None]
    result = {
        "method": method,
        "initialization": None if init is None else str(init),
        "architecture": architecture,
        "tasks": list(bench.tasks),
        "seed": seed,
        "settings": {name: value for name, value in settings.items() if name != "selected"},
        "sweep": sweep,
        "selected": selected,
        "validation_average": max(scores) if scores else None,
        "test": test,
        "seconds": time.monotonic() - started,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30 if on_gpu else None,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(open_clip_state(state), output_dir / "merged.pt")
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_initialization(
    name_or_path: str,
    architecture: str,
    task_count: int,
    assets: ClipAssets,
    device: str,
    runs_dir: Path = Path("runs"),
) -> tuple[dict[str, torch.Tensor], dict]:
    """A merged model to start IF-Merge from: (open_clip visual state dict, metadata).

    ``name_or_path`` is a checkpoint file or a baseline id. For a baseline, the merged
    model of an earlier ``run_baseline`` under ``runs_dir`` is reused; otherwise it is
    rebuilt with the validation-selected hyperparameters stored in
    ``configs/baselines/clip_selected.yaml``, and only if none are stored the full
    selection is run (and stored under ``runs_dir``).
    """
    name = str(name_or_path)
    if name not in BASELINES:
        if not Path(name).is_file():
            raise ValueError(f"{name!r} is neither a baseline id nor a checkpoint file.")
        return _load_state(name), {"checkpoint": name}
    if BASELINES[name].posthoc:
        raise ValueError(f"{name} refines another merged model; pass the path of its merged.pt instead.")
    if name == "wa":
        experts = [visual_state(assets.load_checkpoint(task)) for task in task_names(task_count)]
        return {key: torch.stack([e[key].float() for e in experts]).mean(0) for key in experts[0]}, {
            "method": name
        }

    directory = baseline_directory(runs_dir, architecture, task_count, name)
    if (directory / "merged.pt").is_file():
        return _load_state(directory / "merged.pt"), {
            "method": name,
            "checkpoint": str(directory / "merged.pt"),
        }
    settings = baseline_settings(name, architecture, task_count)
    selected = selected_hyperparameters(name, architecture, task_count)
    if selected is None and ("grid" in settings or name == "mats"):
        result = run_baseline(name, architecture, task_count, assets, device, output_dir=directory)
        return _load_state(directory / "merged.pt"), {"method": name, "selected": result["selected"]}
    bench = _Benchmark(architecture, task_count, assets, device)
    state, selected, _ = _RUNNERS[name](bench, {**settings, "selected": selected or {}}, None)
    return open_clip_state(state), {"method": name, "selected": selected}


# ----------------------------------------------------------------------- CLI
class _ListBaselines(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for method, baseline in BASELINES.items():
            suffix = "  [post-hoc: needs --init]" if baseline.posthoc else ""
            print(f"{method:<11} {baseline.name:<17} {baseline.data}{suffix}")
        parser.exit()


def run(args: argparse.Namespace) -> dict:
    logging.basicConfig(format="%(message)s")
    logger.setLevel(logging.INFO)
    output_dir = baseline_directory(args.runs, args.arch, args.tasks, args.method, args.init)
    assets = ClipAssets(args.arch, args.data)
    result = run_baseline(
        args.method, args.arch, args.tasks, assets, args.device, output_dir, args.init, args.seed
    )
    print(f"[{output_dir.name}] selected {result['selected']}: test accuracy {result['test']['average']:.2f}")
    return result


def register(subparsers) -> None:
    from .run import add_common_arguments

    parser = subparsers.add_parser(
        "baseline",
        help="run a merging baseline on CLIP",
        description="Select the hyperparameters of a baseline on the labeled validation split and "
        "evaluate the selected merged model once on the test split.",
    )
    add_common_arguments(parser)
    parser.add_argument("--method", required=True, choices=list(BASELINES), help="baseline id (see --list)")
    parser.add_argument("--init", help="post-hoc methods: the baseline id or checkpoint path to start from")
    parser.add_argument("--seed", type=int, default=0, help="calibration draw and random seed")
    parser.add_argument(
        "--list", action=_ListBaselines, nargs=0, help="print the baselines and the data they use"
    )
    parser.set_defaults(run=run)
