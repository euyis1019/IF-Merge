"""Linear interpolation paths between merged models (Table S40).

    python analysis/interpolation_paths.py --arch ViT-B-32 --tasks 8 \\
        --models if_wa=runs/.../wa-seed0/merged.pt if_ta=runs/.../ta-seed0/merged.pt \\
                 joint0=runs/.../joint/steps3000-seed0/model.pt \\
        --paths if_ta:if_wa if_ta:joint0

A path is theta(w) = (1 - w) theta_a + w theta_b over all parameters of the visual encoder,
sampled at w = 0, 0.1, ..., 1. Every point is evaluated on 1,024 test images per task (one
fixed draw shared by all paths; ``--images 256`` gives the last worst-task column of the
table), in float32, with the frozen classification heads:

    accuracy barrier            max_w [ chord(w) - accuracy(w) ]  for the task-averaged accuracy
    worst-task accuracy barrier the same per task, maximum over tasks
    cross-entropy barrier       max_w [ CE(w) - chord(w) ]  for the task-averaged cross-entropy
    expert KL                   KL(expert || model), task-averaged: at w = 0, its minimum, at w = 1

where chord(w) interpolates the two end values linearly. A barrier of 0 means that no sampled
point is worse than the chord.
"""

from __future__ import annotations

from common import evaluate_subsets, finish, interpolate, load_models, mean, setting_parser, shared_subsets

from if_merge.clip.assets import ClipAssets
from if_merge.clip.model import visual_state
from if_merge.clip.tasks import task_names

WEIGHTS = [index / 10 for index in range(11)]


def barrier(values: list[float], higher_is_better: bool) -> float:
    """Largest gap between the chord through the end values and the curve, on its worse side."""
    chord = [(1 - w) * values[0] + w * values[-1] for w in WEIGHTS]
    return max((c - v) if higher_is_better else (v - c) for v, c in zip(values, chord, strict=True))


def main() -> None:
    parser = setting_parser(__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--paths", nargs="+", required=True, metavar="A:B")
    parser.add_argument("--images", type=int, default=1024, help="test images per task")
    args = parser.parse_args()
    assets, tasks = ClipAssets(args.arch, args.data), task_names(args.tasks)
    models = load_models(args.models, args, assets)
    subsets, heads = shared_subsets(assets, tasks, args.images)
    teachers = {}
    for task in tasks:
        expert = visual_state(assets.load_checkpoint(task))
        teachers[task] = evaluate_subsets(args.arch, expert, {task: subsets[task]}, heads, args.device)[
            "logits"
        ][task]

    result = {"architecture": args.arch, "tasks": list(tasks), "images_per_task": args.images, "paths": {}}
    for path in args.paths:
        a, b = path.split(":")
        curve = []
        for weight in WEIGHTS:
            state = interpolate(models[a], models[b], weight)
            point = evaluate_subsets(args.arch, state, subsets, heads, args.device, teachers)
            del point["logits"]
            curve.append(point)
        accuracy = [mean(point["accuracy"].values()) for point in curve]
        cross_entropy = [mean(point["cross_entropy"].values()) for point in curve]
        expert_kl = [mean(point["expert_kl"].values()) for point in curve]
        summary = {
            "accuracy_barrier": barrier(accuracy, True),
            "worst_task_accuracy_barrier": max(
                barrier([point["accuracy"][task] for point in curve], True) for task in tasks
            ),
            "cross_entropy_barrier": barrier(cross_entropy, False),
            "expert_kl": {"first": expert_kl[0], "minimum": min(expert_kl), "last": expert_kl[-1]},
            "accuracy": accuracy,
            "cross_entropy": cross_entropy,
            "curve": curve,
        }
        result["paths"][f"{a}|{b}"] = summary
        kl = summary["expert_kl"]
        print(
            f"{a} <-> {b}: accuracy barrier {summary['accuracy_barrier']:.2f} "
            f"(worst task {summary['worst_task_accuracy_barrier']:.2f}), "
            f"cross-entropy barrier {summary['cross_entropy_barrier']:.4f}, "
            f"expert KL {kl['first']:.3f} / {kl['minimum']:.3f} / {kl['last']:.3f}",
            flush=True,
        )
    finish(result, args)


if __name__ == "__main__":
    main()
