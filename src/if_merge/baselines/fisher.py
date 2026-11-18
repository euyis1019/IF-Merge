"""Merging with Fisher information measured at the experts: Fisher Merging and MaTS."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from ..core.solver import conjugate_gradient
from .regmean import State


def diagonal_fisher(model: nn.Module, logit_batches: Iterable[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Diagonal Fisher estimate of ``model`` from batches of its own logits.

    ``logit_batches`` yields the logits of one batch at a time, still attached to the
    graph. Following the FusionBench implementation of Fisher Merging, each batch
    back-propagates ``sum_i sum_c sqrt(p_ic) log p_ic`` once (sqrt(p) held constant) and
    the squared batch gradient is accumulated; the total is divided by the number of
    samples. The estimate therefore depends on the batch size and needs no labels.
    """
    fisher = {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}
    samples = 0
    for logits in logit_batches:
        log_probabilities = logits.log_softmax(dim=-1)
        loss = (log_probabilities.exp().detach().sqrt() * log_probabilities).sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                fisher[name] += parameter.grad.detach() ** 2
        samples += len(logits)
    model.zero_grad(set_to_none=True)
    return {name: (value / samples).cpu() for name, value in fisher.items()}


@torch.no_grad()
def fisher_merge(
    expert_states: Sequence[State],
    fishers: Sequence[Mapping[str, torch.Tensor]],
    floor: float = 1e-6,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    """Fisher Merging: per-parameter average of the experts weighted by their diagonal Fisher.

    ``floor`` is added to every Fisher entry, so parameters without signal fall back to
    a plain average. With ``normalize``, each expert is weighted by the inverse of the
    global L2 norm of its Fisher (weights sum to one), as in FusionBench. Entries without
    a Fisher estimate are averaged.

    Matena and Raffel, "Merging Models with Fisher-Weighted Averaging", NeurIPS 2022.
    """
    weights = torch.ones(len(expert_states))
    if normalize:
        norms = torch.tensor(
            [
                float(torch.sqrt(sum((entry.float() ** 2).sum() for entry in fisher.values())))
                for fisher in fishers
            ]
        )
        inverse = 1.0 / (norms + floor)
        weights = weights * (inverse / inverse.sum())

    merged = {}
    for name, reference in expert_states[0].items():
        if any(name not in fisher for fisher in fishers):
            merged[name] = torch.stack([state[name].float() for state in expert_states]).mean(dim=0)
            continue
        numerator = torch.zeros_like(reference, dtype=torch.float32)
        denominator = torch.zeros_like(numerator)
        for weight, state, fisher in zip(weights.tolist(), expert_states, fishers, strict=True):
            importance = fisher[name].float() + floor
            numerator += weight * importance * state[name].float()
            denominator += weight * importance
        merged[name] = numerator / denominator.clamp(min=1e-12)
    return merged


@torch.no_grad()
def mats(
    expert_states: Sequence[State],
    grams: Sequence[Mapping[str, torch.Tensor]],
    layer_names: Sequence[str],
    initial_state: State,
    iterations: int = 100,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """MaTS with the RegMean objective, solved by conjugate gradient.

    For every ``<layer>.weight`` the normal equations ``sum_t G_t X = sum_t G_t W_t^T`` of
    RegMean are solved with ``iterations`` steps of conjugate gradient started from
    ``initial_state``, without scaling the off-diagonal entries of the Gram matrices.
    Since the solver stops early, the result depends on the initialization. All other
    parameters keep their value in ``initial_state``.

    Tam et al., "Merging by Matching Models in Task Parameter Subspaces", TMLR 2024.
    """
    merged = {name: tensor.clone() for name, tensor in initial_state.items()}
    for layer in layer_names:
        key = f"{layer}.weight"
        task_grams = [task[layer].to(device).float() for task in grams]
        targets = [state[key].to(device).float().transpose(0, 1) for state in expert_states]
        right_hand_side = sum(gram @ target for gram, target in zip(task_grams, targets, strict=True))
        solution = conjugate_gradient(
            lambda x, task_grams=task_grams: sum(gram @ x for gram in task_grams),
            right_hand_side,
            initial_state[key].to(device).float().transpose(0, 1),
            iterations,
            relative_tolerance=1e-8,
        )
        merged[key] = solution.transpose(0, 1).contiguous().cpu()
    return merged
