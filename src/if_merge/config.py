"""The IF-Merge recipe.

One recipe file fixes every choice of the method. The paper uses a single recipe per
architecture (``configs/if_merge/clip.yaml`` and ``configs/if_merge/t5.yaml``); ablations
change one field at a time with ``--set section.field=value``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

A_ESTIMATORS = ("current", "identity")  # measured at the current merged model, or A = I
G_ESTIMATORS = (
    "normalized_kl",  # Eq. (5): normalized teacher-student KL gradients (the paper's G_t)
    "kl",  # the same gradients without the chi-square normalization
    "mc_fisher",  # labels sampled from the merged model's prediction
    "exact_fisher",  # one backward pass per class, merged model's prediction (Gamma_t)
    "empirical_fisher",  # ground-truth label (needs labels; ablation only)
    "output_activation",  # second moment of the layer outputs: a forward statistic, no feedback
    "identity",  # G = I: activation-space fitting
)
REESTIMATE = ("both", "a", "g", "none")


@dataclass(frozen=True)
class ASettings:
    """Activation statistics A_t: second moment of the inputs of each weight matrix."""

    estimator: str = "current"


@dataclass(frozen=True)
class GSettings:
    """Output-side statistics G_t: second moment of the feedback at each layer output."""

    estimator: str = "normalized_kl"
    shrinkage: float | str = "automatic"  # "automatic" (Ledoit-Wolf) or a number in [0, 1]
    mc_samples: int = 4  # only read by estimator = mc_fisher
    # Controls on the task correspondence of G (ablations; the defaults leave G untouched):
    permute_tasks: bool = False  # task t is fitted with the G of task t + 1
    heterogeneity: float = 1.0  # eta in G_t <- eta G_t + (1 - eta) mean_s G_s; 0 shares one G


@dataclass(frozen=True)
class IterationSettings:
    rounds: int = 4
    reestimate: str = "both"  # which statistics are measured again after each update
    relative_update_tolerance: float = 1e-8


@dataclass(frozen=True)
class SolverSettings:
    max_iterations: int = 100
    relative_tolerance: float = 1e-6
    warm_start: str = "current"  # initial guess: "current" weights, "expert_mean" or "zero"


@dataclass(frozen=True)
class SelectionSettings:
    scope: str = "global"  # "global", or "per_block" (global first, then block by block)
    steps: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 1.0)
    # Ablation: round r takes forced_steps[r - 1] in every layer instead of selecting a step;
    # rounds beyond the list select as usual. The endpoint is still chosen by holdout KL.
    forced_steps: tuple[float, ...] = ()


@dataclass(frozen=True)
class CalibrationSettings:
    samples: int = 256  # |C_t|: unlabeled inputs per task for the statistics
    holdout: int = 256  # |H_t|: disjoint unlabeled inputs per task for step selection
    seed: int = 0  # which random draw of C_t and H_t


_SECTIONS = {
    "a": ASettings,
    "g": GSettings,
    "iteration": IterationSettings,
    "solver": SolverSettings,
    "selection": SelectionSettings,
    "calibration": CalibrationSettings,
}


@dataclass(frozen=True)
class IFMergeConfig:
    a: ASettings = field(default_factory=ASettings)
    g: GSettings = field(default_factory=GSettings)
    iteration: IterationSettings = field(default_factory=IterationSettings)
    solver: SolverSettings = field(default_factory=SolverSettings)
    selection: SelectionSettings = field(default_factory=SelectionSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> IFMergeConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_mapping(yaml.safe_load(handle) or {}).with_overrides(overrides)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> IFMergeConfig:
        if not isinstance(raw, dict):
            raise ValueError("The recipe must be a mapping.")
        unknown = set(raw) - set(_SECTIONS)
        if unknown:
            raise ValueError(f"Unknown recipe sections: {sorted(unknown)}")
        values = {}
        for name, constructor in _SECTIONS.items():
            section = dict(raw.get(name) or {})
            if name == "selection":
                for key in ("steps", "forced_steps"):
                    if key in section:
                        section[key] = tuple(float(step) for step in section[key])
            try:
                values[name] = constructor(**section)
            except TypeError as error:
                raise ValueError(f"Invalid recipe section {name!r}: {error}") from error
        config = cls(**values)
        config.validate()
        return config

    def with_overrides(self, overrides: list[str] | tuple[str, ...]) -> IFMergeConfig:
        """Apply ``section.field=value`` overrides, e.g. ``iteration.reestimate=none``."""
        values = self.to_dict()
        for override in overrides:
            key, separator, raw = override.partition("=")
            section, _, name = key.partition(".")
            if not separator or section not in values or name not in values[section]:
                raise ValueError(f"Expected section.field=value with a known field, got {override!r}")
            values[section][name] = yaml.safe_load(raw)
        return self.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["selection"]["steps"] = list(self.selection.steps)
        values["selection"]["forced_steps"] = list(self.selection.forced_steps)
        return values

    def validate(self) -> None:
        choices = [
            ("a.estimator", self.a.estimator, A_ESTIMATORS),
            ("g.estimator", self.g.estimator, G_ESTIMATORS),
            ("iteration.reestimate", self.iteration.reestimate, REESTIMATE),
            ("solver.warm_start", self.solver.warm_start, ("current", "expert_mean", "zero")),
            ("selection.scope", self.selection.scope, ("global", "per_block")),
        ]
        for name, value, allowed in choices:
            if value not in allowed:
                raise ValueError(f"{name} must be one of {list(allowed)}; got {value!r}")
        if self.g.shrinkage != "automatic":
            _number("g.shrinkage", self.g.shrinkage, 0.0, 1.0)
        if type(self.g.permute_tasks) is not bool:
            raise ValueError("g.permute_tasks must be true or false.")
        _number("g.heterogeneity", self.g.heterogeneity, 0.0, 1.0)
        for name, value, minimum in (
            ("g.mc_samples", self.g.mc_samples, 1),
            ("iteration.rounds", self.iteration.rounds, 0),
            ("solver.max_iterations", self.solver.max_iterations, 1),
            ("calibration.samples", self.calibration.samples, 1),
            ("calibration.holdout", self.calibration.holdout, 1),
            ("calibration.seed", self.calibration.seed, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
        _number("solver.relative_tolerance", self.solver.relative_tolerance, 0.0)
        _number("iteration.relative_update_tolerance", self.iteration.relative_update_tolerance, 0.0)
        steps = self.selection.steps
        if 0.0 not in steps or len(set(steps)) != len(steps):
            raise ValueError("selection.steps must be distinct and include 0 (reject the update).")
        for step in steps + self.selection.forced_steps:
            _number("selection.steps", step, 0.0, 1.0)


def _number(name: str, value: Any, minimum: float, maximum: float | None = None) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"at least {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ValueError(f"{name} must be {bound}.")


assert tuple(f.name for f in fields(IFMergeConfig)) == tuple(_SECTIONS)
