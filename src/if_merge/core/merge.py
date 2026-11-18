"""IF-Merge: iterative full-model updates driven by function-space feedback (Alg. 1).

The loop below is independent of the architecture. A *backend* owns the merged model,
the experts and the unlabeled data, and answers three questions about the current
merged model: what are the two-sided statistics, what is the holdout KL, and what are
the expert weights of each layer. ``if_merge.clip`` and ``if_merge.t5`` provide the two
backends used in the paper.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

import torch
import torch.nn as nn

from ..config import IFMergeConfig
from .solver import solve_layers
from .statistics import TaskStatistics, adjust_feedback, combine_statistics


class Backend(Protocol):
    """What IF-Merge needs to know about a model family."""

    device: str
    #: the weight matrices being merged, as live modules of the merged model
    layers: Mapping[str, nn.Linear]
    #: partition of ``layers`` used when the step size is selected per block
    blocks: Mapping[Hashable, Sequence[str]]
    #: layers fitted with G = I whatever the recipe says (e.g. a vocabulary head)
    activation_only: frozenset[str]

    def expert_weights(self, layer: str) -> Sequence[torch.Tensor]:
        """The weight of ``layer`` in every expert, in task order."""

    def collect_statistics(self, config: IFMergeConfig, round_index: int) -> list[TaskStatistics]:
        """A_t and G_t of every task, measured at the current merged model on C_t."""

    def holdout_kl(self) -> float:
        """L_H: the teacher-student KL of the current merged model on H_t, averaged over tasks."""

    def state(self) -> dict[str, torch.Tensor]:
        """A CPU copy of the current merged model."""


@dataclass(frozen=True)
class RoundRecord:
    round: int
    holdout_kl: float
    global_step: float = 0.0
    block_steps: dict[str, float] = field(default_factory=dict)
    relative_update: float = 0.0
    mean_g_shrinkage: float | None = None
    statistics_seconds: float | None = None  # time spent measuring A and G in this round

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IFMergeResult:
    state: dict[str, torch.Tensor]  # the merged model with the lowest holdout KL
    selected_round: int
    rounds: list[RoundRecord]
    stop_reason: str


@torch.no_grad()
def _interpolate(backend: Backend, start, solved, names: Sequence[str], step: float) -> None:
    """Set W <- W_start + step (X - W_start) on the fitted layers among ``names``.

    A backend may ask for this incremental form (``incremental_step = True``); the default is
    the convex form (1 - step) W_start + step X. They are equal in exact arithmetic, but capped
    conjugate gradient leaves poorly determined directions in X where rounding is not negligible,
    so each backend keeps the form its reported runs used.
    """
    incremental = getattr(backend, "incremental_step", False)
    for name in names:
        if name in solved:
            if incremental:
                weight = start[name] + step * (solved[name] - start[name])
            else:
                weight = (1.0 - step) * start[name] + step * solved[name]
            backend.layers[name].weight.copy_(weight)


def _select(
    backend: Backend, start, solved, names: Sequence[str], steps: Sequence[float]
) -> tuple[float, float]:
    """Try every step size on ``names``; return (holdout KL, step), ties going to the smaller step."""
    trials = []
    for step in steps:
        _interpolate(backend, start, solved, names, step)
        trials.append((backend.holdout_kl(), step))
    return min(trials)


def if_merge(
    backend: Backend,
    config: IFMergeConfig,
    on_round: Callable[[RoundRecord], None] | None = None,
) -> IFMergeResult:
    """Refine the backend's merged model in place and return the selected endpoint."""
    steps = tuple(sorted(config.selection.steps))
    layers = backend.layers
    blocks = {key: list(names) for key, names in backend.blocks.items()}

    records = [RoundRecord(round=0, holdout_kl=backend.holdout_kl())]
    best_key, best_state = (records[0].holdout_kl, 0), backend.state()
    if on_round:
        on_round(records[0])

    first_statistics: list[TaskStatistics] | None = None
    stop_reason = "maximum rounds"
    for round_index in range(1, config.iteration.rounds + 1):
        # 1. Two-sided statistics at the current merged model.
        statistics_seconds = None
        if first_statistics is None or config.iteration.reestimate != "none":
            started = time.monotonic()
            fresh = backend.collect_statistics(config, round_index)
            statistics_seconds = time.monotonic() - started
            fresh = adjust_feedback(fresh, config.g.permute_tasks, config.g.heterogeneity)
            statistics = combine_statistics(fresh, first_statistics, config.iteration.reestimate)
            if first_statistics is None and config.iteration.reestimate != "both":
                first_statistics = fresh
            del fresh
        else:
            statistics = first_statistics

        # 2. Fit every weight matrix independently under its two-sided metric.
        start = {name: layer.weight.detach().clone() for name, layer in layers.items()}
        solved, g_shrinkage = solve_layers(
            list(layers),
            backend.expert_weights,
            start,
            statistics,
            config,
            backend.device,
            backend.activation_only,
        )

        # 3. One step size for the full model, selected by holdout KL ...
        forced = config.selection.forced_steps[round_index - 1 : round_index]
        if forced:  # ablation: take a fixed step instead
            global_step = forced[0]
            _interpolate(backend, start, solved, list(layers), global_step)
            holdout_kl = backend.holdout_kl()
        else:
            holdout_kl, global_step = _select(backend, start, solved, list(layers), steps)
            _interpolate(backend, start, solved, list(layers), global_step)
        block_steps = dict.fromkeys(blocks, global_step)
        # ... then, optionally, reselected block by block with the others fixed.
        if config.selection.scope == "per_block" and not forced:
            for key, names in blocks.items():
                _, block_steps[key] = _select(backend, start, solved, names, steps)
                _interpolate(backend, start, solved, names, block_steps[key])
            holdout_kl = backend.holdout_kl()

        update = sum(
            float(((solved[name] - start[name]) * block_steps[key]).square().sum())
            for key, names in blocks.items()
            for name in names
            if name in solved
        )
        norm = sum(float(start[name].square().sum()) for name in solved)
        record = RoundRecord(
            round=round_index,
            holdout_kl=holdout_kl,
            global_step=global_step,
            block_steps={str(key): step for key, step in block_steps.items()},
            relative_update=(update / max(norm, 1e-30)) ** 0.5,
            mean_g_shrinkage=sum(g_shrinkage.values()) / len(g_shrinkage) if g_shrinkage else None,
            statistics_seconds=statistics_seconds,
        )
        records.append(record)
        if (record.holdout_kl, round_index) < best_key:
            best_key, best_state = (record.holdout_kl, round_index), backend.state()
        if on_round:
            on_round(record)

        statistics = start = solved = None  # free this round's tensors before the next measurement
        gc.collect()
        if str(backend.device).startswith("cuda"):
            torch.cuda.empty_cache()
        if all(step == 0.0 for step in block_steps.values()):
            stop_reason = "no step lowers the holdout KL"
            break
        if record.relative_update < config.iteration.relative_update_tolerance:
            stop_reason = "relative update below tolerance"
            break

    return IFMergeResult(best_state, best_key[1], records, stop_reason)
