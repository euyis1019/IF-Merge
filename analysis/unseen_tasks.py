"""Accuracy on the merged tasks and on the tasks outside the merged set (Table S41).

    python analysis/unseen_tasks.py --arch ViT-B-32 --tasks 8 \\
        --models pretrained if_wa=runs/.../wa-seed0/merged.pt \\
                 joint0=runs/.../joint/steps3000-seed0/model.pt \\
        --mix wise_0.7=pretrained:joint0:0.7

For every model, the accuracy averaged over the merged tasks (the ``--tasks`` setting) and
over the unseen tasks: the other tasks of the 20-task benchmark, each classified with its
own zero-shot head (the 12 tasks beyond the 8-task setting in the paper). Each task
contributes 256 test images; the merged and the unseen tasks are drawn with one seeded
generator each, the unseen tasks in alphabetical order, and the encoder runs in float32.
The data root must hold the heads and test images of all 20 tasks
(``if-merge prepare-clip --tasks 20``).

``--mix NAME=A:B:W`` adds the interpolation (1 - W) A + W B of two listed models: the
midpoints of the table (W = 0.5) and the WiSE-FT rows (A = pretrained, B = a jointly trained
model, W = lambda).
"""

from __future__ import annotations

from common import evaluate_subsets, finish, interpolate, load_models, mean, setting_parser, shared_subsets

from if_merge.clip.assets import ClipAssets
from if_merge.clip.tasks import ALL_TASKS, task_names


def main() -> None:
    parser = setting_parser(__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--mix", nargs="+", default=[], metavar="NAME=A:B:W")
    parser.add_argument("--images", type=int, default=256, help="test images per task")
    args = parser.parse_args()
    assets, merged = ClipAssets(args.arch, args.data), task_names(args.tasks)
    unseen = sorted(ALL_TASKS[args.tasks :], key=str.lower)
    models = load_models(args.models, args, assets)
    for spec in args.mix:
        name, _, parts = spec.partition("=")
        a, b, weight = parts.split(":")
        models[name] = interpolate(models[a], models[b], float(weight))

    data = {
        "merged": shared_subsets(assets, merged, args.images),
        "unseen": shared_subsets(assets, unseen, args.images),
    }
    result = {
        "architecture": args.arch,
        "merged_tasks": list(merged),
        "unseen_tasks": unseen,
        "images_per_task": args.images,
        "models": {},
    }
    for name, state in models.items():
        row = {}
        for kind, (subsets, heads) in data.items():
            if subsets:
                accuracy = evaluate_subsets(args.arch, state, subsets, heads, args.device)["accuracy"]
                row[kind] = mean(accuracy.values())
                row[f"{kind}_per_task"] = accuracy
        result["models"][name] = row
        print(
            f"{name:28s} merged {row['merged']:6.2f}   unseen {row.get('unseen', float('nan')):6.2f}",
            flush=True,
        )
    finish(result, args)


if __name__ == "__main__":
    main()
