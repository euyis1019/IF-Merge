"""The layerwise fit of IF-Merge.

Each weight matrix is fitted to the expert weights under the two-sided metric

    X_l = argmin_X  sum_t || A_t^{1/2} (X - W_t) G_t^{1/2} ||_F^2 ,          (Eq. 2)

whose minimizer solves the normal equations  sum_t A_t X G_t = sum_t A_t W_t G_t  (Eq. 6).
Weights are stored as ``nn.Linear`` stores them, [out, in]; internally we solve for the
transpose so that A_t (inputs) multiplies from the left and G_t (outputs) from the right.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch

from ..config import IFMergeConfig, SolverSettings
from .statistics import TaskStatistics


@torch.no_grad()
def conjugate_gradient(
    apply: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    start: torch.Tensor,
    max_iterations: int,
    relative_tolerance: float,
    precondition: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Solve ``apply(x) = target`` for a symmetric positive semidefinite operator."""
    solution = start.clone()
    residual = target - apply(solution)
    target_norm = target.norm().clamp_min(1e-30)
    preconditioned = precondition(residual) if precondition else residual
    direction = preconditioned.clone()
    inner = (residual * preconditioned).sum()
    for _ in range(max_iterations):
        if residual.norm() / target_norm <= relative_tolerance:
            break
        applied = apply(direction)
        curvature = (direction * applied).sum()
        if curvature.abs() < 1e-30:
            break
        step = inner / curvature
        solution += step * direction
        residual -= step * applied
        preconditioned = precondition(residual) if precondition else residual
        next_inner = (residual * preconditioned).sum()
        if inner.abs() < 1e-30:
            break
        direction = preconditioned + next_inner / inner * direction
        inner = next_inner
    return solution


def condition_moment(moment: torch.Tensor, shrinkage: float = 0.0) -> torch.Tensor:
    """Scale normalization (divide by the mean diagonal), then shrinkage toward identity.

    A moment that is exactly zero stays zero: a task whose prediction already equals
    its expert's sends no feedback, and shrinkage must not invent one.
    """
    scale = moment.diagonal().mean()
    if not torch.isfinite(scale) or float(scale.abs()) <= 1e-30:
        return torch.zeros_like(moment)
    moment = moment / scale
    if shrinkage <= 0:
        return moment
    identity = torch.eye(moment.shape[0], device=moment.device, dtype=moment.dtype)
    if shrinkage >= 1:
        return identity
    return (1.0 - shrinkage) * moment + shrinkage * identity


@torch.no_grad()
def solve_layer(
    expert_weights: Sequence[torch.Tensor],
    A: Sequence[torch.Tensor],
    G: Sequence[torch.Tensor] | None,
    current_weight: torch.Tensor,
    solver: SolverSettings,
) -> torch.Tensor | None:
    """Minimize one layer's fitting objective. All tensors are on the same device.

    ``G = None`` fits with G_t = I (activation-space fitting). Returns the fitted
    [out, in] weight, or ``None`` when no task sends feedback to this layer, in which
    case the normal equations read 0 = 0 and the layer keeps its weight.
    """
    targets = [weight.transpose(0, 1) for weight in expert_weights]
    if G is not None:
        active = [index for index, moment in enumerate(G) if bool(torch.count_nonzero(moment))]
        if not active:
            return None
        A = [A[index] for index in active]
        G = [G[index] for index in active]
        targets = [targets[index] for index in active]

    if G is None:
        right_hand_side = sum(a @ w for a, w in zip(A, targets, strict=True))
        diagonal = sum(a.diagonal()[:, None] for a in A).clamp_min(1e-20)

        def apply(x: torch.Tensor) -> torch.Tensor:
            return sum(a @ x for a in A)
    else:
        right_hand_side = sum(a @ w @ g for a, w, g in zip(A, targets, G, strict=True))
        diagonal = sum(
            a.diagonal()[:, None] * g.diagonal()[None, :] for a, g in zip(A, G, strict=True)
        ).clamp_min(1e-20)

        def apply(x: torch.Tensor) -> torch.Tensor:
            return sum(a @ x @ g for a, g in zip(A, G, strict=True))

    if solver.warm_start == "expert_mean":
        start = torch.stack(targets).mean(dim=0)
    elif solver.warm_start == "zero":
        start = torch.zeros_like(targets[0])
    else:
        start = current_weight.transpose(0, 1)
    solution = conjugate_gradient(
        apply,
        right_hand_side,
        start,
        solver.max_iterations,
        solver.relative_tolerance,
        lambda residual: residual / diagonal,  # diagonal preconditioner
    )
    return solution.transpose(0, 1).contiguous()


@torch.no_grad()
def solve_layers(
    layer_names: Sequence[str],
    expert_weights: Callable[[str], Sequence[torch.Tensor]],
    current_weights: Mapping[str, torch.Tensor],
    statistics: Sequence[TaskStatistics],
    config: IFMergeConfig,
    device: str | torch.device,
    activation_only: frozenset[str] = frozenset(),
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Fit every layer independently (step 3 of Alg. 1).

    Layers in ``activation_only`` are fitted with G = I whatever the recipe says.
    Returns the fitted weights (layers without feedback are omitted) and the mean
    shrinkage applied to G per layer.
    """
    solved: dict[str, torch.Tensor] = {}
    mean_g_shrinkage: dict[str, float] = {}
    for name in layer_names:
        if any(name not in task.A for task in statistics):
            continue
        if config.a.estimator == "identity":
            A = [torch.eye(task.A[name].shape[0], device=device) for task in statistics]
        else:
            A = [condition_moment(task.A[name].to(device).float()) for task in statistics]
        G = None
        if config.g.estimator != "identity" and name not in activation_only:
            shrinkages = [
                float(task.G_shrinkage[name] if config.g.shrinkage == "automatic" else config.g.shrinkage)
                for task in statistics
            ]
            G = [
                condition_moment(task.G[name].to(device).float(), shrinkage)
                for task, shrinkage in zip(statistics, shrinkages, strict=True)
            ]
            mean_g_shrinkage[name] = sum(shrinkages) / len(shrinkages)
        weight = solve_layer(
            [w.to(device).float() for w in expert_weights(name)],
            A,
            G,
            current_weights[name].to(device).float(),
            config.solver,
        )
        if weight is None:
            mean_g_shrinkage.pop(name, None)
        else:
            solved[name] = weight
    return solved, mean_g_shrinkage
