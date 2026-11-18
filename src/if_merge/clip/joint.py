"""Joint training on labeled data: the multi-task (MTL) reference of the CLIP benchmark.

    if-merge prepare-clip --arch ViT-B-32 --tasks 14 --train     # caches the training splits
    if-merge joint --arch ViT-B-32 --tasks 14
    if-merge joint --arch ViT-L-14 --tasks 8 --released
    if-merge joint --arch ViT-B-32 --tasks 8 --init iso_c --save-steps 200 500 1000 1500 2000

The pretrained visual encoder is fine-tuned on the training splits of all tasks at once, each
task under its frozen zero-shot head, and evaluated like every merged model. The validation
images that select the hyperparameters of the baselines are excluded from training.

Recipe (3,000 steps for every backbone and task set): AdamW, learning rate 1e-5, weight decay
0.1, 500 warm-up steps followed by a cosine schedule, gradient clipping at norm 1, batches of
128 images drawn evenly from the tasks, bfloat16 autocast. Training enables TF32 matrix
products, and the trained model is evaluated in the same process: evaluating the saved
``model.pt`` again with ``if-merge evaluate`` (TF32 off) can differ on a few test images.

``--init`` starts from a merged model instead of the pretrained weights, and ``--save-steps``
also writes the model after the given steps (``step<N>.pt``); the endpoint analyses use both.

``--released`` does not train: it downloads the jointly trained 8-task checkpoint released
with FusionBench (ViT-B/32 and ViT-L/14 only) and evaluates it with this benchmark's heads
and test sets. It needs the ``huggingface-hub`` and ``safetensors`` packages.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .assets import ClipAssets, normalize_images
from .evaluate import evaluate, head_logits, head_to
from .model import State, build_exposed_visual, open_clip_state, visual_state
from .run import add_common_arguments, initial_state, setting_dir
from .tasks import task_names

WEIGHT_DECAY, WARMUP_STEPS, GRADIENT_CLIP = 0.1, 500, 1.0
# Images per forward pass; the gradients of one batch are accumulated over its micro-batches.
MICRO_BATCH = {"ViT-B-32": 128, "ViT-B-16": 64, "ViT-L-14": 32}
# Jointly trained 8-task checkpoints released with FusionBench (Hugging Face CLIPVisionModel weights).
RELEASED = {
    "ViT-B-32": "tanganke/clip-vit-base-patch32_TA8_mtl",
    "ViT-L-14": "tanganke/clip-vit-large-patch14_TA8_mtl",
}


class TrainingImages(Dataset):
    """The training splits of all tasks without their validation images, read one batch at a time.

    The caches are memory-mapped: the training splits of the 20-task benchmark take about 150 GB.
    """

    def __init__(self, assets: ClipAssets, tasks: Sequence[str]) -> None:
        self.images, self.labels, self.kept = [], [], []
        for task in tasks:
            path = assets.image_path(task, "train")
            if not path.is_file():
                raise FileNotFoundError(f"{path} is missing; run `if-merge prepare-clip --train` first.")
            stored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            images, labels, validation = stored["x"], stored["y"], stored["val_idx"].long()
            if (
                images.dtype != torch.uint8
                or tuple(images.shape[1:]) != (3, 224, 224)
                or len(images) != len(labels)
            ):
                raise ValueError(f"{task}/train: unexpected image cache")
            keep = torch.ones(len(labels), dtype=torch.bool)
            keep[validation] = False
            if int(keep.sum()) != len(labels) - len(validation):
                raise ValueError(f"{task}/train: the validation indices are not distinct")
            self.images.append(images)
            self.labels.append(labels)
            self.kept.append(keep.nonzero().flatten())
        self.lengths = [len(kept) for kept in self.kept]
        self.offsets = torch.tensor([0, *itertools.accumulate(self.lengths)])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Images, labels and task ids of a batch of global indices, in the order given."""
        index = torch.as_tensor(indices)
        task_ids = torch.bucketize(index, self.offsets[1:], right=True)
        images = torch.empty(len(index), 3, 224, 224, dtype=torch.uint8)
        labels = torch.empty(len(index), dtype=torch.long)
        for task_id in task_ids.unique().tolist():
            rows = (task_ids == task_id).nonzero().flatten()
            source = self.kept[task_id][index[rows] - self.offsets[task_id]]
            images[rows] = self.images[task_id][source]
            labels[rows] = self.labels[task_id][source]
        return images, labels, task_ids


class BalancedBatches(Sampler):
    """``steps`` batches with the same number of images from every task.

    Each task is visited in its own random order, reshuffled whenever it is exhausted.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, steps: int, seed: int) -> None:
        if batch_size < len(lengths):
            raise ValueError("The batch size must be at least the number of tasks.")
        self.lengths, self.batch_size, self.steps, self.seed = list(lengths), batch_size, steps, seed
        self.offsets = [0, *itertools.accumulate(self.lengths)]

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed)
        tasks = len(self.lengths)
        quota = [self.batch_size // tasks + (task < self.batch_size % tasks) for task in range(tasks)]
        orders: list[torch.Tensor | None] = [None] * tasks
        cursors = [0] * tasks
        for _ in range(self.steps):
            parts = []
            for task, (count, length) in enumerate(zip(quota, self.lengths, strict=True)):
                while count:
                    if orders[task] is None or cursors[task] == length:
                        orders[task], cursors[task] = torch.randperm(length, generator=generator), 0
                    taken = min(count, length - cursors[task])
                    parts.append(orders[task][cursors[task] : cursors[task] + taken] + self.offsets[task])
                    cursors[task] += taken
                    count -= taken
            batch = torch.cat(parts)
            yield batch[torch.randperm(len(batch), generator=generator)].tolist()


def learning_rate_at(step: int, base: float, total_steps: int, warmup: int = WARMUP_STEPS) -> float:
    """Linear warm-up, then a cosine schedule that ends at ``total_steps``; ``step`` counts from 0."""
    if step < warmup:
        return base * (step + 1) / warmup
    if total_steps <= warmup:
        return base
    progress = min(max((step - warmup) / (total_steps - warmup), 0.0), 1.0)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    architecture: str,
    tasks: Sequence[str],
    assets: ClipAssets,
    steps: int = 3000,
    seed: int = 0,
    device: str = "cuda",
    batch_size: int = 128,
    micro_batch: int | None = None,
    learning_rate: float = 1e-5,
    workers: int = 8,
    initialization: State | None = None,
    save_steps: Sequence[int] = (),
    directory: Path | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Fine-tune a visual encoder on all tasks; returns its state and a training log.

    Training starts from the pretrained weights unless ``initialization`` is given. The model
    after each of ``save_steps`` is written to ``directory``.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # stays on for the evaluation that follows
    torch.backends.cudnn.allow_tf32 = True
    cuda = str(device).startswith("cuda")
    micro_batch = micro_batch or MICRO_BATCH[architecture]

    start = assets.load_checkpoint(None) if initialization is None else initialization
    visual = build_exposed_visual(architecture, start).to(device)
    heads = [head_to(assets.load_head(task), device) for task in tasks]
    data = TrainingImages(assets, tasks)
    sampler = BalancedBatches(data.lengths, batch_size, steps, seed)
    options = {"persistent_workers": True, "prefetch_factor": 4} if workers else {}
    loader = DataLoader(
        data, sampler=sampler, batch_size=None, num_workers=workers, pin_memory=cuda, **options
    )

    parameters = list(visual.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    seen = torch.zeros(len(tasks), dtype=torch.long)
    visual.train()
    started, running_loss, log = time.monotonic(), 0.0, []
    for step, (images, labels, task_ids) in enumerate(loader, start=1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step - 1, learning_rate, steps)
        seen += torch.bincount(task_ids, minlength=len(tasks))
        images = normalize_images(images, device)
        labels, task_ids = labels.to(device), task_ids.to(device)
        for start in range(0, len(labels), micro_batch):
            rows = slice(start, start + micro_batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                features = visual(images[rows])
            loss = features.new_zeros((), dtype=torch.float32)
            for task_id, head in enumerate(heads):
                members = (task_ids[rows] == task_id).nonzero().flatten()
                if len(members):
                    logits = head_logits(features.index_select(0, members), head)
                    loss = loss + F.cross_entropy(
                        logits, labels[rows].index_select(0, members), reduction="sum"
                    )
            (loss / len(labels)).backward()
            running_loss += float(loss.detach()) / len(labels)
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step in save_steps:
            torch.save(open_clip_state(visual), directory / f"step{step}.pt")
        if step % 100 == 0 or step == steps:
            interval = step % 100 or 100
            minutes = (time.monotonic() - started) / 60
            print(f"step {step}/{steps}: loss {running_loss / interval:.4f} ({minutes:.1f} min)", flush=True)
            log.append({"step": step, "loss": running_loss / interval, "minutes": minutes})
            running_loss = 0.0
    record = {
        "steps": steps,
        "seed": seed,
        "batch_size": batch_size,
        "micro_batch": micro_batch,
        "learning_rate": learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "gradient_clip": GRADIENT_CLIP,
        "training_images": dict(zip(tasks, data.lengths, strict=True)),
        "images_seen": dict(zip(tasks, seen.tolist(), strict=True)),
        "log": log,
        "training_minutes": (time.monotonic() - started) / 60,
    }
    return open_clip_state(visual), record


def convert_released(released: State, pretrained: State) -> dict[str, torch.Tensor]:
    """Hugging Face ``CLIPVisionModel`` weights -> open_clip visual state dict.

    ``CLIPVisionModel`` has no output projection, so the projection of ``pretrained`` (an
    open_clip visual state dict) is kept: it is the one every expert was fine-tuned with.
    """
    source = "vision_model."
    state = {
        "class_embedding": released[f"{source}embeddings.class_embedding"],
        "conv1.weight": released[f"{source}embeddings.patch_embedding.weight"],
        "positional_embedding": released[f"{source}embeddings.position_embedding.weight"],
        "ln_pre.weight": released[f"{source}pre_layrnorm.weight"],
        "ln_pre.bias": released[f"{source}pre_layrnorm.bias"],
        "ln_post.weight": released[f"{source}post_layernorm.weight"],
        "ln_post.bias": released[f"{source}post_layernorm.bias"],
        "proj": pretrained["proj"],
    }
    renamed = {
        "attn.out_proj": "self_attn.out_proj",
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
    }
    block = 0
    while f"{source}encoder.layers.{block}.layer_norm1.weight" in released:
        theirs, ours = f"{source}encoder.layers.{block}.", f"transformer.resblocks.{block}."
        for kind in ("weight", "bias"):
            projections = [released[f"{theirs}self_attn.{name}_proj.{kind}"] for name in "qkv"]
            state[f"{ours}attn.in_proj_{kind}"] = torch.cat(projections)
            for name, their_name in renamed.items():
                state[f"{ours}{name}.{kind}"] = released[f"{theirs}{their_name}.{kind}"]
        block += 1
    if state.keys() != pretrained.keys() or any(state[key].shape != pretrained[key].shape for key in state):
        raise ValueError("The released checkpoint does not match the architecture.")
    return {key: value.float() for key, value in state.items()}


def released_state(architecture: str, assets: ClipAssets) -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    repository = RELEASED[architecture]
    folder = assets.clip / "downloads" / repository.split("/")[1]
    path = hf_hub_download(repository, "model.safetensors", local_dir=folder)
    return convert_released(load_file(path), visual_state(assets.load_checkpoint(None)))


def run(args: argparse.Namespace) -> dict:
    assets, tasks = ClipAssets(args.arch, args.data), task_names(args.tasks)
    if args.released and (args.arch not in RELEASED or args.tasks != 8):
        raise SystemExit("Released checkpoints exist for ViT-B-32 and ViT-L-14 with 8 tasks only.")
    missing = assets.missing(tasks, splits=("test",))
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} files, e.g. {missing[0]}. Run `if-merge prepare-clip` first."
        )
    name = args.name or ("released" if args.released else f"steps{args.steps}-seed{args.seed}")
    if args.init and not args.name:
        name = re.sub(r"[^\w.+-]", "_", Path(args.init).stem) + f"-{name}"
    output = setting_dir(args.runs, args.arch, args.tasks) / "joint" / name
    if (output / "result.json").exists() and not args.force:
        print(f"[skip] {output / 'result.json'} exists (use --force to run again)")
        return json.loads((output / "result.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    cuda = str(args.device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(args.device)
    if args.released:
        state, source = released_state(args.arch, assets), {"released_checkpoint": RELEASED[args.arch]}
    else:
        initialization = None
        if args.init:
            initialization, _ = initial_state(
                args.init, args.arch, args.tasks, assets, args.device, Path(args.runs)
            )
        state, source = train(
            args.arch,
            tasks,
            assets,
            args.steps,
            args.seed,
            args.device,
            args.batch_size,
            args.micro_batch,
            args.learning_rate,
            args.workers,
            initialization,
            args.save_steps,
            output,
        )
        source["initialization"] = args.init or "pretrained"
    scores = evaluate(args.arch, state, assets, tasks, args.device)
    print(f"[{name}] {scores['average']:.2f}", flush=True)
    record = {
        "architecture": args.arch,
        "tasks": list(tasks),
        "after": scores,
        **source,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(args.device) / 2**30 if cuda else None,
    }
    if not args.no_save:
        torch.save(state, output / "model.pt")
    (output / "result.json").write_text(json.dumps(record, indent=2))
    return record


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "joint",
        help="joint training on labeled data (the MTL reference)",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0, help="order of the training images")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=None, help="images per forward pass (memory only)")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=8, help="data-loading processes")
    parser.add_argument(
        "--init",
        default=None,
        help="start from a baseline id or a checkpoint instead of the pretrained weights",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        nargs="*",
        default=[],
        metavar="STEP",
        help="also write the model after these steps",
    )
    parser.add_argument(
        "--released", action="store_true", help="evaluate the released 8-task checkpoint instead"
    )
    parser.add_argument(
        "--name", default=None, help="run name (default: [<init>-]steps<N>-seed<K>, or `released`)"
    )
    parser.add_argument("--no-save", action="store_true", help="do not write model.pt")
    parser.add_argument("--force", action="store_true", help="run even if the result exists")
    parser.set_defaults(run=run)
