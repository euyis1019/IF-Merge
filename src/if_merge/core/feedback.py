"""The function-space feedback signal: normalized teacher-student KL."""

from __future__ import annotations

import torch

EPSILON = 1e-8


def feedback_loss(teacher_logits: torch.Tensor, logits: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Per-sample teacher-student KL, divided by sqrt(chi-square + eps) (Eq. 5).

    The denominator is held fixed, so the gradient with respect to the logits is
    (q - p) / sqrt(d + eps): backpropagating the sum yields the feedback u of every layer.
    """
    teacher = teacher_logits.float().softmax(dim=-1).detach()
    log_student = logits.float().log_softmax(dim=-1)
    student = log_student.exp()
    tiny = torch.finfo(torch.float32).tiny
    kl = (teacher * (teacher.clamp_min(tiny).log() - log_student)).sum(dim=-1)
    if not normalize:
        return kl
    chi_square = ((teacher - student).square() / student.clamp_min(tiny)).sum(dim=-1)
    return kl / (chi_square + EPSILON).sqrt().detach()
