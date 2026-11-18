"""Merging through the singular structure of task matrices: Iso-C, Iso-CTS, TSV-M and WUDI.

These methods treat every two-dimensional parameter as a task matrix and merge it on its
own; all other parameters are averaged. Which tensors are two-dimensional is decided by
the layout of the state dict the task vectors come from (see ``docs/BASELINES.md`` for the
CLIP attention projections).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .weight_space import TaskVector


def _mean(task_vectors: Sequence[TaskVector], name: str) -> torch.Tensor:
    return torch.stack([tv[name] for tv in task_vectors]).mean(dim=0)


@torch.no_grad()
def iso_c(task_vectors: Sequence[TaskVector], device: str = "cuda") -> dict[str, torch.Tensor]:
    """Iso-C: the Task Arithmetic matrix with all singular values replaced by their mean.

    Marczak et al., "No Task Left Behind: Isotropic Model Merging with Common and
    Task-Specific Subspaces", ICML 2025.
    """
    output = {}
    for name, reference in task_vectors[0].items():
        stacked = torch.stack([tv[name].to(device) for tv in task_vectors])
        if reference.ndim != 2:
            output[name] = stacked.mean(dim=0).cpu()
            continue
        left, values, right = torch.linalg.svd(stacked.sum(dim=0).float(), full_matrices=False)
        isotropic = torch.full_like(values, values.mean())
        output[name] = (left @ torch.diag(isotropic) @ right).to(reference.dtype).cpu()
    return output


@torch.no_grad()
def iso_cts(
    task_vectors: Sequence[TaskVector], common_fraction: float = 0.8, device: str = "cuda"
) -> dict[str, torch.Tensor]:
    """Iso-CTS: Iso-C with part of the spectrum given to task-specific directions.

    The top ``common_fraction`` of the singular directions of the Task Arithmetic matrix
    form the common subspace. The remaining rank is shared equally between the tasks and
    filled with the top singular directions of each task matrix after projecting out the
    common subspace. The combined bases are orthogonalized and given one singular value.

    Marczak et al., ICML 2025 (see ``iso_c``).
    """
    task_count = len(task_vectors)
    output = {}
    for name, reference in task_vectors[0].items():
        if reference.ndim != 2:
            output[name] = _mean(task_vectors, name)
            continue
        matrices = [tv[name].to(device).float() for tv in task_vectors]
        rows, columns = matrices[0].shape
        rank = min(rows, columns)
        specific_total = round((rank - int(rank * common_fraction)) / task_count) * task_count
        common, specific = rank - specific_total, specific_total // task_count

        common_left, common_values, common_right = torch.linalg.svd(sum(matrices), full_matrices=False)
        common_left, common_values = common_left[:, :common], common_values[:common]
        common_right = common_right[:common]
        left = torch.zeros(rows, rank, device=device)
        values = torch.zeros(rank, device=device)
        right = torch.zeros(rank, columns, device=device)
        for index, matrix in enumerate(matrices):
            residual = matrix - common_left @ (common_left.transpose(0, 1) @ matrix)
            task_left, task_values, task_right = torch.linalg.svd(residual, full_matrices=False)
            block = slice(index * specific, (index + 1) * specific)
            left[:, block] = task_left[:, :specific]
            values[block] = task_values[:specific]
            right[block] = task_right[:specific]
        left[:, task_count * specific :] = common_left
        values[task_count * specific :] = common_values
        right[task_count * specific :] = common_right

        isotropic = torch.full_like(values, values.mean())
        merged = _orthogonalize(left) @ torch.diag(isotropic) @ _orthogonalize(right)
        output[name] = merged.to(reference.dtype).cpu()
    return output


@torch.no_grad()
def tsv_m(task_vectors: Sequence[TaskVector], device: str = "cuda") -> dict[str, torch.Tensor]:
    """TSV-M: keep the top 1/T singular directions of every task matrix and decorrelate them.

    The retained left and right singular vectors of all tasks are concatenated and
    orthogonalized (the closest orthogonal bases); the singular values are kept.

    Gargiulo et al., "Task Singular Vectors: Reducing Task Interference in Model Merging",
    CVPR 2025.
    """
    task_count = len(task_vectors)
    output = {}
    for name, reference in task_vectors[0].items():
        if reference.ndim != 2:
            output[name] = _mean(task_vectors, name)
            continue
        matrices = [tv[name].to(device).float() for tv in task_vectors]
        rows, columns = matrices[0].shape
        rank = min(rows, columns)
        retained = rank // task_count
        left = torch.zeros(rows, rank, device=device)
        values = torch.zeros(rank, device=device)
        right = torch.zeros(rank, columns, device=device)
        for index, matrix in enumerate(matrices):
            task_left, task_values, task_right = torch.linalg.svd(matrix, full_matrices=False)
            block = slice(index * retained, (index + 1) * retained)
            left[:, block] = task_left[:, :retained]
            values[block] = task_values[:retained]
            right[block] = task_right[:retained]
        right_outer, _, right_inner = torch.linalg.svd(right, full_matrices=False)
        merged = _orthogonalize(left) @ torch.diag(values) @ right_outer @ right_inner
        output[name] = merged.to(reference.dtype).cpu()
    return output


def _orthogonalize(basis: torch.Tensor) -> torch.Tensor:
    """The closest matrix with orthonormal columns (or rows): U V^T of the SVD."""
    outer, _, inner = torch.linalg.svd(basis, full_matrices=False)
    return outer @ inner


def wudi(
    task_vectors: Sequence[TaskVector], steps: int = 300, learning_rate: float = 1e-5, device: str = "cuda"
) -> dict[str, torch.Tensor]:
    """WUDI: data-free minimization of the interference between task matrices.

    Starting from the Task Arithmetic matrix, Adam minimizes, for every matrix M,
    ``sum_t ||(M - tau_t) tau_t^T||_F^2 / ||tau_t||_F^2``: the merged matrix should act
    like ``tau_t`` on the row space of ``tau_t``. The optimization is deterministic.

    Cheng et al., "Whoever Started the Interference Should End It: Guiding Data-Free Model
    Merging via Task Vectors", ICML 2025.
    """
    output = {}
    for name, reference in task_vectors[0].items():
        stacked = torch.stack([tv[name].to(device).float() for tv in task_vectors])
        if reference.ndim != 2:
            output[name] = stacked.mean(dim=0).to(reference.dtype).cpu()
            continue
        merged = torch.nn.Parameter(stacked.sum(dim=0).clone())
        optimizer = torch.optim.Adam([merged], lr=learning_rate, weight_decay=0.0)
        squared_norms = torch.linalg.vector_norm(stacked.reshape(len(stacked), -1), dim=-1).square()
        with torch.enable_grad():
            for _ in range(steps):
                projected = (merged.unsqueeze(0) - stacked) @ stacked.transpose(1, 2)
                loss = torch.sum(projected.square() / squared_norms[:, None, None])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        output[name] = merged.detach().to(reference.dtype).cpu()
    return output
