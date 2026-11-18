"""Weight distance and input-weighted distance between merged models (Table S35).

    python analysis/parameter_distances.py --arch ViT-B-32 --tasks 8 \\
        --models wa=wa ta=ta if_wa=runs/.../wa-seed0/merged.pt if_ta=runs/.../ta-seed0/merged.pt

Two distances over the six weight matrices (query, key, value and output projection of the
attention, the two MLP maps) of Transformer blocks 2 to the last:

    weights          sqrt( sum_l || W_l - W'_l ||_F^2 )
    input-weighted   sqrt( sum_l tr( (W_l - W'_l) A_l (W_l - W'_l)^T ) )

where A_l is the second moment of the inputs of layer l in the *pretrained* model, measured
on 64 test images per task (fixed seed) over all tokens. Both are divided by the mean
distance from the pretrained model to an expert under the same measure, so 1 is one expert
fine-tuning distance. The patch embedding, the output projection and blocks 0 and 1 are left
out (``--first-block``).

The script prints the distance of every pair of the given models (or of ``--pairs A:B``),
of every model to ``pretrained``, and the median and range over the pairs of experts. The
rows of Table S35 are medians and ranges of such pairs: initializations among each other,
their IF-Merge endpoints among each other, an initialization and its own endpoint (IF-Merge
or distillation), endpoints of the same initialization under two calibration seeds, and
endpoints against jointly trained models.
"""

from __future__ import annotations

import itertools
import re
import statistics

import torch
from common import finish, linear_weights, load_models, load_state, setting_parser, test_subset

from if_merge.clip.assets import ClipAssets, normalize_images
from if_merge.clip.model import build_exposed_visual, linear_layers, visual_state
from if_merge.clip.tasks import task_names


def block_of(name: str) -> int | None:
    match = re.match(r"transformer\.resblocks\.(\d+)\.", name)
    return int(match.group(1)) if match else None


@torch.no_grad()
def input_moments(args, assets: ClipAssets, tasks, images: int) -> dict[str, torch.Tensor]:
    """Second moment of the inputs of every Transformer weight matrix in the pretrained model."""
    visual = build_exposed_visual(args.arch, assets.load_checkpoint(None)).to(args.device)
    layers = {name: layer for name, layer in linear_layers(visual).items() if block_of(name) is not None}
    sums: dict[str, torch.Tensor] = {}
    rows = dict.fromkeys(layers, 0)

    def record(name: str):
        def hook(_module, inputs, _output):
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).double()
            sums[name] = sums.get(name, 0) + x.T @ x
            rows[name] += len(x)

        return hook

    handles = [layer.register_forward_hook(record(name)) for name, layer in layers.items()]
    for task in tasks:  # the generator is seeded anew for every task
        subset, _ = test_subset(assets, task, images, torch.Generator().manual_seed(7))
        visual(normalize_images(subset, args.device))
    for handle in handles:
        handle.remove()
    return {name: (sums[name] / rows[name]).float() for name in layers}


@torch.no_grad()
def distances(a, b, moments, blocks, device) -> tuple[float, float]:
    """(weight distance, input-weighted distance) over the matrices of ``blocks``, not normalized."""
    plain, weighted = dict.fromkeys(blocks, 0.0), dict.fromkeys(blocks, 0.0)
    for name in sorted(moments):
        block = block_of(name)
        if block in plain:
            difference = (a[name] - b[name]).to(device)
            plain[block] += float((difference * difference).sum())
            weighted[block] += float(((difference @ moments[name]) * difference).sum())
    return sum(plain.values()) ** 0.5, sum(weighted.values()) ** 0.5


def main() -> None:
    parser = setting_parser(__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--pairs", nargs="+", default=None, metavar="A:B", help="default: all pairs")
    parser.add_argument("--first-block", type=int, default=2)
    parser.add_argument("--images", type=int, default=64, help="test images per task for the input moments")
    args = parser.parse_args()
    assets, tasks = ClipAssets(args.arch, args.data), task_names(args.tasks)
    moments = input_moments(args, assets, tasks, args.images)
    blocks = range(args.first_block, 1 + max(block_of(name) for name in moments))

    weights = {"pretrained": linear_weights(args.arch, load_state("pretrained", args, assets))}
    experts = [f"expert:{task}" for task in tasks]
    for name, task in zip(experts, tasks, strict=True):
        weights[name] = linear_weights(args.arch, visual_state(assets.load_checkpoint(task)))
    models = load_models(args.models, args, assets)
    for name, state in models.items():
        weights[name] = linear_weights(args.arch, state)

    def raw(a: str, b: str) -> tuple[float, float]:
        return distances(weights[a], weights[b], moments, blocks, args.device)

    to_experts = [raw("pretrained", expert) for expert in experts]
    unit = [sum(values) / len(values) for values in zip(*to_experts, strict=True)]

    def normalized(a: str, b: str) -> dict[str, float]:
        plain, weighted = raw(a, b)
        return {"weights": plain / unit[0], "input_weighted": weighted / unit[1]}

    between_experts = [normalized(a, b) for a, b in itertools.combinations(experts, 2)]
    summary = {
        measure: {
            "median": statistics.median(pair[measure] for pair in between_experts),
            "min": min(pair[measure] for pair in between_experts),
            "max": max(pair[measure] for pair in between_experts),
        }
        for measure in ("weights", "input_weighted")
    }
    print(f"between experts ({len(between_experts)} pairs): {summary}")
    if args.pairs:
        pairs = [tuple(pair.split(":")) for pair in args.pairs]
    else:
        pairs = list(itertools.combinations(models, 2)) + [("pretrained", name) for name in models]
    result_pairs = {}
    for a, b in pairs:
        result_pairs[f"{a}|{b}"] = normalized(a, b)
        print(
            f"{a} | {b}: weights {result_pairs[f'{a}|{b}']['weights']:.3f}  "
            f"input-weighted {result_pairs[f'{a}|{b}']['input_weighted']:.3f}"
        )
    result = {
        "architecture": args.arch,
        "tasks": list(tasks),
        "blocks": list(blocks),
        "unit": {"weights": unit[0], "input_weighted": unit[1]},
        "between_experts": summary,
        "pairs": result_pairs,
    }
    finish(result, args)


if __name__ == "__main__":
    main()
