"""Spectral bands of the differences between merged models (Fig. 3, Tables S36 and S37). CPU only.

    python analysis/spectral_bands.py --arch ViT-B-32 --tasks 8 \\
        --models wa=wa:runs/.../wa-seed0/merged.pt iso_c=iso_c:runs/.../iso_c-seed0/merged.pt

A model is given as ``NAME=INITIALIZATION:ENDPOINT``, two checkpoints. For each of the weight matrices
IF-Merge fits (query, key and value projections separately: 74 matrices for ViT-B), the sum
of the expert task increments, sum_t (W_t - W_pre) = U diag(s) V^T, defines a frame, and a
band is a range of singular indices: top 1%, 1-10%, 10-50%, bottom 50% (also top 10% and
bottom 90%). A matrix X restricted to a band is the block of U^T X V whose rows and columns
both lie in the band.

Band projection coefficient (Table S36, Fig. 3), for a pair of models a, b:

    rho = < D_band, S_band > / || S_band ||_F^2 ,

with S the difference of the two initializations and D the difference of the two endpoints;
1 keeps the initial difference along its own direction, 0 removes it. The reported value is
the median over matrices; a matrix is skipped for a band that holds less than 1e-6 of the
energy of S. The endpoint may come from any method: IF-Merge, distillation (the KD columns)
or joint training started from the two initializations (the joint-training row).
``random_bands`` repeats the computation after randomly permuting the singular pairs before
banding (mean over ``--permutations`` of the per-permutation median): the reference row.
Fig. 3 shows the coefficients of the pairs that involve ``wa``.

Band magnitude spread (Table S37): for every model, the mean diagonal entry of
U^T (W - W_pre) V inside a band, divided by mean(s), median over matrices; the spread is the
ratio of the largest to the smallest such value over the models named by ``--spread``, for the
initializations and for the endpoints.
"""

from __future__ import annotations

import itertools
import statistics

import torch
from common import finish, linear_weights, load_state, setting_parser

from if_merge.clip.assets import ClipAssets
from if_merge.clip.model import visual_state
from if_merge.clip.tasks import task_names

BANDS = {
    "top 1%": (0.0, 0.01),
    "1-10%": (0.01, 0.10),
    "10-50%": (0.10, 0.50),
    "bottom 50%": (0.50, 1.0),
    "top 10%": (0.0, 0.10),
    "bottom 90%": (0.10, 1.0),
}
MINIMUM_SHARE = 1e-6


def band_slices(rank: int) -> dict[str, slice]:
    slices = {}
    for name, (low, high) in BANDS.items():
        start = int(round(low * rank))
        slices[name] = slice(start, max(int(round(high * rank)), start + 1))
    return slices


def coefficient(initial: torch.Tensor, final: torch.Tensor, rows, floor: float = 0.0) -> float | None:
    """rho of one band; ``rows`` selects the band's rows and columns in the singular frame."""
    s, d = initial[rows][:, rows], final[rows][:, rows]
    energy = float((s * s).sum())
    return float((d * s).sum()) / energy if energy > 0 and energy >= floor else None


def median(values) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def main() -> None:
    parser = setting_parser(__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=INITIALIZATION:ENDPOINT")
    parser.add_argument("--pairs", nargs="+", default=None, metavar="A:B", help="default: all pairs")
    parser.add_argument("--spread", nargs="+", default=None, metavar="NAME", help="default: all models")
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    args.device = "cpu"
    torch.set_num_threads(args.threads)
    assets, tasks = ClipAssets(args.arch, args.data), task_names(args.tasks)

    pretrained = linear_weights(args.arch, load_state("pretrained", args, assets))
    names = sorted(pretrained)
    tau = {name: torch.zeros_like(pretrained[name], dtype=torch.float64) for name in names}
    for task in tasks:
        expert = linear_weights(args.arch, visual_state(assets.load_checkpoint(task)))
        for name in names:
            tau[name] += expert[name].double() - pretrained[name].double()

    weights: dict[str, dict[str, torch.Tensor]] = {}  # checkpoint -> matrices
    runs = {}
    for spec in args.models:
        run, _, checkpoints = spec.partition("=")
        runs[run] = tuple(checkpoints.split(":"))
        for checkpoint in runs[run]:
            if checkpoint not in weights:
                weights[checkpoint] = linear_weights(args.arch, load_state(checkpoint, args, assets))
    pairs = [tuple(p.split(":")) for p in args.pairs] if args.pairs else list(itertools.combinations(runs, 2))
    members = args.spread or list(runs)

    rho = {pair: {band: [] for band in BANDS} for pair in pairs}
    random_rho = {pair: [{band: [] for band in BANDS} for _ in range(args.permutations)] for pair in pairs}
    profiles = {checkpoint: {band: [] for band in BANDS} for checkpoint in weights}
    generator = torch.Generator().manual_seed(0)
    for name in names:
        U, s, Vh = torch.linalg.svd(tau[name], full_matrices=False)
        slices = band_slices(s.numel())
        base = pretrained[name].double()
        frame = {c: U.T @ (matrices[name].double() - base) @ Vh.T for c, matrices in weights.items()}
        for checkpoint, coordinates in frame.items():
            diagonal = torch.diagonal(coordinates)
            for band, rows in slices.items():
                profiles[checkpoint][band].append(float(diagonal[rows].mean()) / float(s.mean()))
        permutations = [torch.randperm(s.numel(), generator=generator) for _ in range(args.permutations)]
        for a, b in pairs:
            initial = frame[runs[a][0]] - frame[runs[b][0]]
            final = frame[runs[a][1]] - frame[runs[b][1]]
            floor = MINIMUM_SHARE * float((initial * initial).sum())
            for band, rows in slices.items():
                rho[(a, b)][band].append(coefficient(initial, final, rows, floor))
                for index, permutation in enumerate(permutations):
                    random_rho[(a, b)][index][band].append(coefficient(initial, final, permutation[rows]))
        print(f"{name}: rank {s.numel()}", flush=True)

    def spread(position: int) -> dict[str, float]:
        values = {band: [median(profiles[runs[run][position]][band]) for run in members] for band in BANDS}
        return {band: max(v) / max(min(v), 1e-12) for band, v in values.items()}

    result = {
        "architecture": args.arch,
        "tasks": list(tasks),
        "matrices": len(names),
        "pairs": {f"{a}|{b}": {band: median(v) for band, v in rho[(a, b)].items()} for a, b in pairs},
        "random_bands": {
            f"{a}|{b}": {
                band: statistics.mean(median(permutation[band]) for permutation in random_rho[(a, b)])
                for band in BANDS
            }
            for a, b in pairs
            if args.permutations
        },
        "spread": {"models": members, "initializations": spread(0), "endpoints": spread(1)},
    }
    for pair, bands in result["pairs"].items():
        print(pair, " ".join(f"{band}: {value:+.2f}" for band, value in bands.items() if value is not None))
    print("spread", result["spread"])
    finish(result, args)


if __name__ == "__main__":
    main()
