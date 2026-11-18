"""ReForge: anchor-regularized regression refined by Bayesian optimization.

Li et al., "ReForge: Refining Merged Models with Anchor-Regularized Regression",
arXiv:2605.12843. This is a reimplementation that approximates the method: the inner
solve is the paper's closed form, the outer search uses our own 10-dimensional
parameterization instead of the paper's 15 dimensions (see ``docs/BASELINES.md``).

Inner solve, per weight matrix W ([out, in]), with expert input Gram matrices G_t and an
anchor U0 taken from an existing merged model:

    W = (sum_t W_t G_t + lambda U0) (sum_t G_t + lambda I)^-1 ,

i.e. RegMean with a ridge that pulls toward the anchor. The Gram matrices are measured
once; the outer search only re-solves this closed form.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Mapping, Sequence

import torch

from .regmean import State, expert_mean


@torch.no_grad()
def anchored_regression(
    pretrained: State,
    expert_states: Sequence[State],
    grams: Sequence[Mapping[str, torch.Tensor]],
    layer_names: Sequence[str],
    anchor: State,
    log_ridge: Sequence[float],
    off_diagonal_scale: Sequence[float],
    output_scale: Sequence[float],
    group_of: Callable[[str], int],
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Solve every ``<layer>.weight`` in closed form; other parameters are the expert mean.

    The three hyperparameters are indexed by the group of the layer (``group_of``):
    ``log_ridge`` is log10 of lambda relative to the mean diagonal of ``sum_t G_t``,
    ``off_diagonal_scale`` multiplies the off-diagonal entries of each Gram matrix (as in
    RegMean), and ``output_scale`` rescales the solved task vector ``W - W_0``.
    """
    merged = expert_mean(expert_states)
    for layer in layer_names:
        key, group = f"{layer}.weight", group_of(layer)
        scale = off_diagonal_scale[group]
        numerator = denominator = None
        for state, task_grams in zip(expert_states, grams, strict=True):
            gram = task_grams[layer].to(device).float()
            gram = gram * scale + torch.diag_embed(gram.diagonal() * (1.0 - scale))
            weight = state[key].to(device).float()
            numerator = weight @ gram if numerator is None else numerator + weight @ gram
            denominator = gram if denominator is None else denominator + gram
        size = denominator.shape[0]
        ridge = (10.0 ** log_ridge[group]) * float(denominator.diagonal().sum()) / size
        system = denominator + ridge * torch.eye(size, device=device, dtype=denominator.dtype)
        target = numerator + ridge * anchor[key].to(device).float()
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info) == 0:
            solution = torch.cholesky_solve(target.transpose(0, 1).contiguous(), factor).transpose(0, 1)
        else:
            solution = target @ torch.linalg.pinv(system)
        base = pretrained[key].to(device).float()
        merged[key] = (base + output_scale[group] * (solution - base)).contiguous().cpu()
    return merged


def depth_groups(
    layer_names: Sequence[str], groups: int = 3, pattern: str = r"resblocks\.(\d+)\."
) -> Callable:
    """Map layer names to ``groups`` consecutive depth ranges of equal size (shallow to deep)."""
    blocks = {name: int(re.search(pattern, name).group(1)) for name in layer_names}
    first, span = min(blocks.values()), max(blocks.values()) - min(blocks.values()) + 1

    def group_of(name: str) -> int:
        return min(groups - 1, int((blocks[name] - first) * groups / span))

    return group_of


class GaussianProcessOptimizer:
    """Bayesian optimization (maximization) with a Gaussian-process surrogate.

    The first ``initial`` points are uniform random; afterwards each proposal maximizes
    the expected improvement over ``candidates`` uniform random points. Requires
    scikit-learn and scipy.
    """

    def __init__(
        self, bounds: Sequence[tuple[float, float]], seed: int = 0, initial: int = 20, candidates: int = 4096
    ):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        self.low = torch.tensor([bound[0] for bound in bounds], dtype=torch.float64)
        self.high = torch.tensor([bound[1] for bound in bounds], dtype=torch.float64)
        self.generator = torch.Generator().manual_seed(seed)
        self.initial, self.candidates = initial, candidates
        self.points: list[list[float]] = []
        self.scores: list[float] = []
        kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0] * len(bounds), nu=2.5) + WhiteKernel(1e-4)
        self.process = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed
        )

    def _sample(self, count: int) -> torch.Tensor:
        uniform = torch.rand(count, len(self.low), generator=self.generator, dtype=torch.float64)
        return self.low + uniform * (self.high - self.low)

    def ask(self) -> list[float]:
        if len(self.points) < self.initial:
            return self._sample(1)[0].tolist()
        import numpy as np
        from scipy.stats import norm
        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():  # kernel hyperparameters that end at their bounds
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.process.fit(np.array(self.points), np.array(self.scores))
        candidates = self._sample(self.candidates).numpy()
        mean, deviation = self.process.predict(candidates, return_std=True)
        deviation = np.maximum(deviation, 1e-9)
        improvement = mean - max(self.scores) - 0.01
        z = improvement / deviation
        expected = improvement * norm.cdf(z) + deviation * norm.pdf(z)
        return candidates[int(np.argmax(expected))].tolist()

    def tell(self, point: Sequence[float], score: float) -> None:
        self.points.append(list(point))
        self.scores.append(float(score))

    def best(self) -> tuple[list[float], float]:
        index = max(range(len(self.scores)), key=lambda i: self.scores[i])
        return self.points[index], self.scores[index]
