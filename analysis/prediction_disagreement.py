"""Top-1 prediction disagreement between merged models (Table S34).

    python analysis/prediction_disagreement.py --arch ViT-B-32 --tasks 8 \\
        --models wa=runs/.../wa-seed0/merged.pt ta=runs/.../ta-seed0/merged.pt iso_c=... \\
        --group ordinary=wa,ta --group spectral=iso_c

For every pair of models, the share (%) of test images on which their top-1 predictions
differ, averaged over tasks. Each task contributes 1,024 test images, drawn once with a
fixed seed and shared by all models; the encoder runs under float16 autocast as in
``if-merge evaluate``. ``--group`` averages the pairs inside a group and between two groups.

Table S34 is two runs per setting: one over the initializations (baseline ids as
checkpoints) and one over their IF-Merge endpoints. "Ordinary, initial" is the group
ordinary|ordinary of the first run; the other columns are groups of the second run
(ordinary|ordinary, ordinary|spectral, the pair iso_c|iso_cts, ordinary|subspace,
spectral|subspace). Ordinary: wa, ta, ties, regmean, adamerging, regmeanpp; spectral: iso_c,
iso_cts; subspace: tsv_m, wudi. Initializations that are the same model are counted once
(on ViT-B/32 with 20 tasks, ta equals wa). The averages quoted in the main text (14.8% among
initializations, 1.9% among endpoints) are the means of the two ordinary|ordinary columns
over the nine settings.
"""

from __future__ import annotations

import itertools

import torch
from common import finish, load_models, logits, setting_parser, test_subset

from if_merge.clip.assets import ClipAssets
from if_merge.clip.evaluate import head_to
from if_merge.clip.model import build_exposed_visual
from if_merge.clip.tasks import task_names


def main() -> None:
    parser = setting_parser(__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--group", action="append", default=[], metavar="LABEL=NAME,NAME")
    parser.add_argument("--images", type=int, default=1024, help="test images per task")
    args = parser.parse_args()
    assets, tasks = ClipAssets(args.arch, args.data), task_names(args.tasks)
    models = load_models(args.models, args, assets)
    groups = {label: names.split(",") for label, _, names in (g.partition("=") for g in args.group)}
    unknown = {name for names in groups.values() for name in names} - set(models)
    if unknown:
        parser.error(f"groups name unknown models: {sorted(unknown)}")

    # One fixed subset per task: the generator is seeded anew for every task, indices sorted.
    subsets = {
        task: test_subset(assets, task, args.images, torch.Generator().manual_seed(0), sort=True)
        for task in tasks
    }
    heads = {task: head_to(assets.load_head(task), args.device) for task in tasks}
    predictions, accuracy = {}, {}
    for name, state in models.items():
        visual = build_exposed_visual(args.arch, state).to(args.device)
        predictions[name], correct = {}, []
        for task in tasks:
            images, labels = subsets[task]
            scores = logits(visual, images, heads[task], args.device, half=True, batch_size=256)
            predictions[name][task] = scores.softmax(dim=-1).argmax(dim=-1).cpu()
            correct.append(float((predictions[name][task] == labels).float().mean()) * 100)
        accuracy[name] = sum(correct) / len(correct)
        print(f"{name:24s} accuracy on the subset {accuracy[name]:6.2f}", flush=True)

    disagreement = {}
    for a, b in itertools.combinations(models, 2):
        per_task = [float((predictions[a][t] != predictions[b][t]).float().mean()) * 100 for t in tasks]
        disagreement[f"{a}|{b}"] = sum(per_task) / len(per_task)
        print(f"{a} | {b}: {disagreement[f'{a}|{b}']:.2f}")

    def between(a: str, b: str) -> float:
        return disagreement.get(f"{a}|{b}", disagreement.get(f"{b}|{a}"))

    group_means = {}
    for g, h in itertools.combinations_with_replacement(groups, 2):
        pairs = itertools.combinations(groups[g], 2) if g == h else itertools.product(groups[g], groups[h])
        values = [between(a, b) for a, b in pairs if a != b]
        if values:
            group_means[f"{g}|{h}"] = sum(values) / len(values)
            print(f"group {g} | {h}: {group_means[f'{g}|{h}']:.2f} ({len(values)} pairs)")
    result = {
        "architecture": args.arch,
        "tasks": list(tasks),
        "images_per_task": args.images,
        "accuracy": accuracy,
        "disagreement": disagreement,
        "groups": group_means,
    }
    finish(result, args)


if __name__ == "__main__":
    main()
