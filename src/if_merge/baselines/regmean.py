"""Activation-space regression: RegMean and RegMean++.

Each weight matrix W (stored [out, in] as in ``nn.Linear``) is the closed-form solution of

    min_W  sum_t || (W - W_t) X_t ||_F^2      =>      W^T = (sum_t G_t)^-1 sum_t G_t W_t^T ,

where G_t = X_t X_t^T is the Gram matrix of the layer inputs of task t. RegMean measures
the inputs in the experts; RegMean++ feeds each expert layer with the features produced
by the merged model up to that point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

State = Mapping[str, torch.Tensor]


def expert_mean(expert_states: Sequence[State]) -> dict[str, torch.Tensor]:
    return {
        name: torch.stack([s[name].float() for s in expert_states]).mean(dim=0) for name in expert_states[0]
    }


def scale_off_diagonal(gram: torch.Tensor, scale: float) -> torch.Tensor:
    """Multiply the off-diagonal entries of a Gram matrix by ``scale`` (RegMean's regularizer)."""
    mask = torch.full_like(gram, scale)
    mask.fill_diagonal_(1.0)
    return gram * mask


@torch.no_grad()
def regmean(
    expert_states: Sequence[State],
    grams: Sequence[Mapping[str, torch.Tensor]],
    layer_names: Sequence[str],
    off_diagonal_scale: float = 0.1,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """RegMean: closed-form regression for ``<layer>.weight``, the expert mean elsewhere.

    ``grams[t][layer]`` is the Gram matrix of the inputs of ``layer`` in expert t on its
    own task. Raises ``torch.linalg.LinAlgError`` if the summed Gram matrix is singular.

    Jin et al., "Dataless Knowledge Fusion by Merging Weights of Language Models", ICLR 2023.
    """
    merged = expert_mean(expert_states)
    for layer in layer_names:
        key = f"{layer}.weight"
        numerator = denominator = None
        for state, task_grams in zip(expert_states, grams, strict=True):
            gram = scale_off_diagonal(task_grams[layer].to(device).float(), off_diagonal_scale)
            weight = state[key].to(device).float().transpose(0, 1)  # [in, out]
            numerator = gram @ weight if numerator is None else numerator + gram @ weight
            denominator = gram if denominator is None else denominator + gram
        merged[key] = torch.linalg.solve(denominator, numerator).transpose(0, 1).contiguous().cpu()
    return merged


@torch.no_grad()
def regmean_plus_plus(
    blocks: Sequence[nn.Module],
    expert_block_states: Sequence[Sequence[State]],
    block_inputs: list[torch.Tensor],
    scratch_block: nn.Module,
    off_diagonal_scale: float,
    batch_size: int = 32,
) -> None:
    """RegMean++: merge a stack of blocks in one forward pass, in place.

    ``blocks`` are the blocks of the merged model, ``expert_block_states[t][b]`` is the
    state dict of block b of expert t, and ``block_inputs[t]`` are the inputs of the first
    block for the calibration data of task t, computed with the merged model (on the
    device of the blocks). For every block, the inputs produced by the merged prefix are
    sent through each expert's block to collect the Gram matrices of its ``nn.Linear``
    inputs; the linear weights are merged as in RegMean (float64 solve), every other
    parameter of the block is averaged, and the merged block advances the inputs.
    ``scratch_block`` is a block of the same architecture used to run the expert blocks.

    Nguyen et al., "RegMean++: Enhancing Effectiveness and Generalization of Regression
    Mean for Model Merging", TMLR 2026.
    """
    task_count = len(block_inputs)
    device = block_inputs[0].device
    linear_names = [name for name, module in scratch_block.named_modules() if isinstance(module, nn.Linear)]

    def batches(inputs: torch.Tensor):
        return (inputs[start : start + batch_size] for start in range(0, len(inputs), batch_size))

    for index, block in enumerate(blocks):
        grams: dict[str, list[torch.Tensor]] = {name: [] for name in linear_names}
        for task in range(task_count):
            scratch_block.load_state_dict(expert_block_states[task][index], strict=True)
            sums: dict[str, torch.Tensor] = {}
            counts = dict.fromkeys(linear_names, 0)

            def record(name: str, sums=sums, counts=counts):
                def hook(_module, inputs, _output):
                    rows = inputs[0].reshape(-1, inputs[0].shape[-1]).double()
                    gram = rows.transpose(0, 1) @ rows
                    sums[name] = gram if name not in sums else sums[name] + gram
                    counts[name] += rows.shape[0]

                return hook

            handles = [scratch_block.get_submodule(n).register_forward_hook(record(n)) for n in linear_names]
            for batch in batches(block_inputs[task]):
                scratch_block(batch)
            for handle in handles:
                handle.remove()
            for name in linear_names:
                grams[name].append(sums[name] / counts[name])

        merged_state = {}
        for name in linear_names:
            left, right = 0.0, 0.0
            for task in range(task_count):
                gram = grams[name][task]
                diagonal = torch.diag(torch.diagonal(gram))
                gram = off_diagonal_scale * gram + (1.0 - off_diagonal_scale) * diagonal
                weight = expert_block_states[task][index][f"{name}.weight"].to(device).double()
                left = left + gram
                right = right + gram @ weight.transpose(0, 1)
            merged_state[f"{name}.weight"] = torch.linalg.solve(left, right).transpose(0, 1).float()
        for key in expert_block_states[0][index]:
            if key not in merged_state:  # biases and normalization layers: expert average
                values = [states[index][key].float() for states in expert_block_states]
                merged_state[key] = sum(values).div(task_count)
        block.load_state_dict({key: value.to(device) for key, value in merged_state.items()}, strict=True)
        for task in range(task_count):
            block_inputs[task] = torch.cat([block(batch) for batch in batches(block_inputs[task])])
