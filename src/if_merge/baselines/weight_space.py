"""Weight-space merging: Weight Averaging, Task Arithmetic and TIES.

Every function maps the task vectors ``tau_t = theta_t - theta_0`` of the experts to one
merged task vector; the merged model is ``theta_0 + scale * merged``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

TaskVector = Mapping[str, torch.Tensor]


def weight_average(task_vectors: Sequence[TaskVector]) -> dict[str, torch.Tensor]:
    """Mean of the task vectors; with scale 1 this is the average of the expert weights.

    Wortsman et al., "Model Soups", ICML 2022.
    """
    return {name: torch.stack([tv[name] for tv in task_vectors]).mean(dim=0) for name in task_vectors[0]}


def task_arithmetic(task_vectors: Sequence[TaskVector]) -> dict[str, torch.Tensor]:
    """Sum of the task vectors.

    Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023.
    """
    return {name: torch.stack([tv[name] for tv in task_vectors]).sum(dim=0) for name in task_vectors[0]}


def ties(task_vectors: Sequence[TaskVector], retained_percentage: float = 20.0) -> dict[str, torch.Tensor]:
    """TIES: trim, elect sign, disjoint merge.

    Each task vector keeps its ``retained_percentage`` largest-magnitude entries, ranked
    over the whole flattened vector. The sign of every parameter is elected by the sum of
    the trimmed entries, and the entries that agree with it are summed.

    Yadav et al., "TIES-Merging: Resolving Interference When Merging Models", NeurIPS 2023.
    """
    if not 0 < retained_percentage < 100:
        raise ValueError("retained_percentage must lie strictly between 0 and 100.")
    names = list(task_vectors[0])
    flat = torch.stack([torch.cat([tv[name].reshape(-1) for name in names]) for tv in task_vectors])
    retained = int(flat.shape[1] * retained_percentage / 100.0)
    magnitude = flat.abs()
    threshold = magnitude.kthvalue(flat.shape[1] - retained, dim=1, keepdim=True).values
    trimmed = flat * (magnitude > threshold)
    elected_sign = torch.sign(trimmed.sum(dim=0))
    agrees = (torch.sign(trimmed) == elected_sign.unsqueeze(0)) & (trimmed != 0)
    merged = (trimmed * agrees).sum(dim=0)

    output, offset = {}, 0
    for name in names:
        tensor = task_vectors[0][name]
        output[name] = merged[offset : offset + tensor.numel()].reshape(tensor.shape)
        offset += tensor.numel()
    return output
