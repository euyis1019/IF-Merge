"""Mixed initializations for the spectral-transplant experiment (Table 8a, Table S38a,b). CPU only.

    python analysis/mix_initializations.py --arch ViT-B-32 --tasks 8 \\
        --a wa --b iso_c --blocks 4 6 8 9 10 11 --out-dir runs/mixed
    if-merge clip --arch ViT-B-32 --tasks 8 --init runs/mixed/wa_head+iso_c_tail.pt

Builds initializations that combine two merged models A and B by spectral band, and writes
them as open_clip visual state dicts that ``if-merge clip --init`` refines.

Each weight matrix has a frame given by the sum of the expert task increments,
sum_t (W_t - W_pre) = U diag(s) V^T. The head H is the leading 10% of the singular pairs
(``round(0.1 r)``), the tail T the rest, and P_H(X) = U_H U_H^T X V_H V_H^T (P_T alike). The
matrices are those of the open_clip state dict: the query, key and value projections of a
block form one packed matrix here (50 matrices for ViT-B), unlike in ``spectral_bands.py``,
where they are three (74 matrices).

    A_head+B_tail.pt     W_pre + P_H(A - W_pre) + P_T(B - W_pre)       "WA-H + Iso-C-T"
    B_head+A_tail.pt     W_pre + P_H(B - W_pre) + P_T(A - W_pre)       "Iso-C-H + WA-T"
    A_head+random.pt     W_pre + P_H(A - W_pre) + c Q_L Q_R^T          "WA-H + Random"
    A_linear+B_other.pt  weight matrices of A, all other parameters of B
    B_linear+A_other.pt  weight matrices of B, all other parameters of A
    A+B_blocks_<..>.pt   A with the weight matrices of the ``--blocks`` taken from B

Q_L and Q_R have Haar-random orthonormal columns, as many as the tail has directions, and c
gives the random part the Frobenius norm of P_T(B - W_pre): a flat spectrum with the energy
of B's tail, unrelated to the task directions. In the first three files all parameters that
are not weight matrices (biases, LayerNorms, embeddings) are those of A.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch
from common import load_models, load_state, setting_parser

from if_merge.clip.assets import ClipAssets
from if_merge.clip.model import visual_state
from if_merge.clip.tasks import task_names

HEAD_FRACTION = 0.10


def weight_matrices(state) -> list[str]:
    """Keys of the weight matrices, block by block, then patch embedding and output projection."""
    depth = 1 + max(
        int(m.group(1)) for m in (re.match(r"transformer\.resblocks\.(\d+)\.", k) for k in state) if m
    )
    parts = ("attn.in_proj_weight", "attn.out_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
    return [f"transformer.resblocks.{block}.{part}" for block in range(depth) for part in parts] + [
        "conv1.weight",
        "proj",
    ]


def as_matrix(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """[out, in] float64 view of a weight: the patch embedding is flattened, ``proj`` transposed."""
    tensor = tensor.double()
    return tensor.flatten(1) if key == "conv1.weight" else tensor.T if key == "proj" else tensor


def from_matrix(key: str, matrix: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    matrix = matrix.T if key == "proj" else matrix
    return matrix.reshape(like.shape).float().contiguous()


def haar(rows: int, columns: int, generator: torch.Generator) -> torch.Tensor:
    """``columns`` orthonormal columns in R^rows, Haar distributed."""
    q, r = torch.linalg.qr(torch.randn(rows, columns, generator=generator, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def main() -> None:
    parser = setting_parser(__doc__)
    parser.add_argument("--a", required=True, metavar="NAME=CHECKPOINT", help="e.g. wa")
    parser.add_argument("--b", required=True, metavar="NAME=CHECKPOINT", help="e.g. iso_c")
    parser.add_argument("--blocks", type=int, nargs="*", default=[], help="blocks taken from B in A+B_blocks")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random part")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    args.device = "cpu"
    assets, tasks = ClipAssets(args.arch, args.data), task_names(args.tasks)
    (name_a, a), (name_b, b) = load_models([args.a, args.b], args, assets).items()
    pretrained = load_state("pretrained", args, assets)
    experts = [visual_state(assets.load_checkpoint(task)) for task in tasks]
    keys = weight_matrices(pretrained)

    mixed = {name: dict(a) for name in ("head_tail", "tail_head", "head_random")}
    for index, key in enumerate(keys):
        base = as_matrix(key, pretrained[key])
        tau = sum(as_matrix(key, expert[key]) - base for expert in experts)
        U, s, Vh = torch.linalg.svd(tau, full_matrices=False)
        head = int(round(HEAD_FRACTION * s.numel()))
        tail = s.numel() - head

        def project(matrix, part, U=U, V=Vh.T, head=head):
            left, right = (U[:, :head], V[:, :head]) if part == "head" else (U[:, head:], V[:, head:])
            return left @ (left.T @ matrix @ right) @ right.T

        delta_a, delta_b = as_matrix(key, a[key]) - base, as_matrix(key, b[key]) - base
        tail_b = project(delta_b, "tail")
        generator = torch.Generator().manual_seed(args.seed * 100003 + index)
        q_left, q_right = haar(base.shape[0], tail, generator), haar(base.shape[1], tail, generator)
        random = float(tail_b.norm()) / math.sqrt(tail) * (q_left @ q_right.T)
        parts = {
            "head_tail": project(delta_a, "head") + tail_b,
            "tail_head": project(delta_b, "head") + project(delta_a, "tail"),
            "head_random": project(delta_a, "head") + random,
        }
        for name, part in parts.items():
            mixed[name][key] = from_matrix(key, base + part, pretrained[key])
        print(f"{key}: {tuple(base.shape)}, head {head} of {s.numel()} directions", flush=True)

    files = {
        f"{name_a}_head+{name_b}_tail": mixed["head_tail"],
        f"{name_b}_head+{name_a}_tail": mixed["tail_head"],
        f"{name_a}_head+random": mixed["head_random"],
        f"{name_a}_linear+{name_b}_other": {k: (a if k in keys else b)[k].clone() for k in a},
        f"{name_b}_linear+{name_a}_other": {k: (b if k in keys else a)[k].clone() for k in a},
    }
    if args.blocks:
        chosen = tuple(f"transformer.resblocks.{block}." for block in args.blocks)
        label = "-".join(str(block) for block in args.blocks)
        files[f"{name_a}+{name_b}_blocks_{label}"] = {
            k: (b if k in keys and k.startswith(chosen) else a)[k].clone() for k in a
        }
    directory = Path(args.out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, state in files.items():
        torch.save(state, directory / f"{name}.pt")
        print(f"wrote {directory / f'{name}.pt'}")


if __name__ == "__main__":
    main()
