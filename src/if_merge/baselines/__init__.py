"""Merging baselines compared with IF-Merge in the paper.

The algorithms are functions on task vectors, state dicts and statistics, independent of
the architecture; ``if_merge.clip.baselines`` runs them on the CLIP benchmark with the
paper's protocol. ``docs/BASELINES.md`` lists, per method, the reference implementation
it follows and where it deviates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Baseline:
    name: str
    #: what the method uses besides the pretrained and expert weights; the labeled
    #: validation split is only used to select hyperparameters
    data: str
    #: refines an existing merged model, which has to be given as its initialization
    posthoc: bool = False


_VALIDATION = "labeled validation split"
_CALIBRATION = "256 unlabeled calibration images per task"

BASELINES: dict[str, Baseline] = {
    "wa": Baseline("Weight Averaging", "none"),
    "ta": Baseline("Task Arithmetic", _VALIDATION),
    "ties": Baseline("TIES", _VALIDATION),
    "tsv_m": Baseline("TSV-M", _VALIDATION),
    "wudi": Baseline("WUDI", _VALIDATION),
    "iso_c": Baseline("Iso-C", _VALIDATION),
    "iso_cts": Baseline("Iso-CTS", _VALIDATION),
    "regmean": Baseline("RegMean", f"{_CALIBRATION}, {_VALIDATION}"),
    "regmeanpp": Baseline("RegMean++", f"{_CALIBRATION}, {_VALIDATION}"),
    "fisher": Baseline("Fisher Merging", _CALIBRATION),
    "mats": Baseline("MaTS", f"{_CALIBRATION}, {_VALIDATION} (for its Task Arithmetic start)"),
    "adamerging": Baseline("AdaMerging", "512 unlabeled calibration images per task"),
    "lines": Baseline("LiNeS", _VALIDATION, posthoc=True),
    "svc": Baseline("SVC", _VALIDATION, posthoc=True),
    "reforge": Baseline("ReForge", f"128 unlabeled calibration images per task, {_VALIDATION}", posthoc=True),
}
