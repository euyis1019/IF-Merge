"""Post-hoc rescaling of a merged task vector: LiNeS and SVC."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .weight_space import TaskVector


def lines(
    merged_task_vector: TaskVector, block_of: Callable[[str], int | None], depth_ratio: float = 1.0
) -> dict[str, torch.Tensor]:
    """LiNeS: scale the merged task vector linearly with depth.

    A tensor of block l (of L) is multiplied by ``1 + depth_ratio * l / (L - 1)``; tensors
    outside the blocks (``block_of`` returns None) are unchanged. With the global scaling
    coefficient ``alpha`` applied afterwards, this is the schedule ``alpha + beta * l / (L - 1)``
    of the paper with ``depth_ratio = beta / alpha``.

    Wang et al., "LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model
    Merging", ICLR 2025.
    """
    indices = {name: block_of(name) for name in merged_task_vector}
    last = max(index for index in indices.values() if index is not None)
    if last < 1:
        raise ValueError("LiNeS needs at least two blocks.")
    return {
        name: tensor if indices[name] is None else tensor * (1.0 + depth_ratio * (indices[name] / last))
        for name, tensor in merged_task_vector.items()
    }


@dataclass
class _Factors:
    left: torch.Tensor
    values: torch.Tensor
    right: torch.Tensor
    #: [rank, tasks]: how much of each task's response along a singular direction the
    #: merged matrix carries (1 = exactly the task's own response)
    ratios: torch.Tensor
    dtype: torch.dtype


class SingularValueCalibration:
    """SVC: rescale the singular values of a merged task matrix, keeping its singular vectors.

    For the merged matrix ``M = U S V^T`` and every left singular vector u_r, the projected
    responses ``u_r^T M`` and ``u_r^T tau_t`` give the ratio
    ``s_rt = <u_r^T M, u_r^T tau_t> / ||u_r^T tau_t||^2``; a ratio above one means the
    direction is over-counted. The singular value is multiplied by
    ``gamma_r = T / sum_t max(strength, s_rt)``. Other tensors are left as they are.

    The scaling coefficient of the merged task vector has to be applied before the
    calibration, because the ratios scale with it. The decomposition is computed once
    (float64) and ``apply`` evaluates any (scale, strength) from it.

    "When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging",
    arXiv:2602.05536.
    """

    def __init__(
        self, task_vectors: Sequence[TaskVector], merged_task_vector: TaskVector, device: str = "cuda"
    ) -> None:
        self.task_count = len(task_vectors)
        self.factors: dict[str, torch.Tensor | _Factors] = {}
        with torch.no_grad():
            for name, tensor in merged_task_vector.items():
                if tensor.ndim != 2:
                    self.factors[name] = tensor
                    continue
                merged = tensor.to(device).double()
                tasks = torch.stack([tv[name].to(device).double() for tv in task_vectors])
                left, values, right = torch.linalg.svd(merged, full_matrices=False)
                directions = left.transpose(0, 1).contiguous()  # [rank, rows]
                merged_response = directions @ merged
                task_responses = torch.einsum("rm,kmn->rkn", directions, tasks)
                norms = (task_responses * task_responses).sum(dim=-1).clamp_min(1e-12)
                ratios = torch.einsum("rn,rkn->rk", merged_response, task_responses) / norms
                self.factors[name] = _Factors(
                    left.cpu(),
                    values.cpu(),
                    right.cpu(),
                    torch.nan_to_num(ratios, nan=0.0).cpu(),
                    tensor.dtype,
                )

    @torch.no_grad()
    def apply(self, scale: float, strength: float) -> dict[str, torch.Tensor]:
        """The calibrated version of ``scale * merged_task_vector``."""
        output = {}
        for name, factors in self.factors.items():
            if isinstance(factors, torch.Tensor):
                output[name] = factors * scale
                continue
            gamma = self.task_count / torch.clamp(factors.ratios * scale, min=strength).sum(dim=1)
            calibrated = (factors.left * (gamma * factors.values * scale).unsqueeze(0)) @ factors.right
            output[name] = calibrated.to(factors.dtype)
        return output
