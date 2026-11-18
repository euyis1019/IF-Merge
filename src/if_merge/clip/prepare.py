"""Download and preprocess everything the CLIP benchmark needs.

    if-merge prepare-clip --arch ViT-B-32 --tasks 8
    if-merge prepare-clip --arch ViT-B-32 ViT-B-16 ViT-L-14 --tasks 20 --seeds 0 1 2

Sources
    expert checkpoints and zero-shot heads   nik-dim/tall_masks (Hugging Face)
    images                                   one Hugging Face dataset per task (see tasks.py)

Splits (per task)
    test    the official test split; the only split results are reported on
    val     10% of the training split (at most 5,000 images), labeled, used only to
            select the hyperparameters of baselines
    calib   unlabeled training images outside `val`; draw K uses the permutation of
            numpy's default_rng(K). IF-Merge reads the first |C_t| images for the
            statistics and the next |H_t| for step selection.
    train   the full training split (only for the jointly trained reference; --train)
"""

from __future__ import annotations

import argparse
import gzip
import io
import pickle
import sys
import types
from glob import glob
from pathlib import Path

import numpy as np
import torch

from .assets import ClipAssets, default_data_root
from .tasks import ALL_TASKS, ARCHITECTURES, TASKS, task_names

CHECKPOINT_REPOSITORY = "nik-dim/tall_masks"
# directory of each backbone inside the checkpoint repository
REPOSITORY_ARCHITECTURE = {"ViT-B-32": "ViT-B-32", "ViT-B-16": "VIT-B-16", "ViT-L-14": "VIT-L-14"}

VAL_FRACTION, MAX_VAL, SPLIT_SEED = 0.1, 5000, 0
# The experts were fine-tuned with the labels in this order, which differs from the class
# names their zero-shot head was built from. Reproducing the benchmark means keeping it.
FER2013_LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


# ------------------------------------------------------------------ checkpoints
def prepare_checkpoints(assets: ClipAssets, tasks) -> None:
    from huggingface_hub import hf_hub_download

    folder = f"single_task/{REPOSITORY_ARCHITECTURE[assets.architecture]}"

    downloads = assets.clip / "downloads"

    def fetch(remote: str, target: Path) -> None:
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        source = hf_hub_download(CHECKPOINT_REPOSITORY, f"{folder}/{remote}", local_dir=downloads)
        Path(source).replace(target)
        print(f"checkpoint  {target}", flush=True)

    fetch("MNISTVal/nonlinear_zeroshot.pt", assets.checkpoint_path(None))
    for task in tasks:
        fetch(f"{task}Val/nonlinear_finetuned.pt", assets.checkpoint_path(task))
        head = assets.head_path(task)
        if not head.exists():
            head.parent.mkdir(parents=True, exist_ok=True)
            source = hf_hub_download(
                CHECKPOINT_REPOSITORY, f"{folder}/head_{task}Val.pt", local_dir=downloads
            )
            torch.save(_load_head(source), head)
            print(f"head        {head}", flush=True)


def _load_head(path: str) -> dict:
    """Read a pickled ``ClassificationHead`` module without the repository that defined it."""
    module = types.ModuleType("src.models.modeling")
    module.ClassificationHead = type("ClassificationHead", (torch.nn.Linear,), {})
    for name in ("src", "src.models"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["src.models.modeling"] = module
    try:
        head = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        sys.modules.pop("src.models.modeling", None)
    state = head.state_dict()
    return {
        "weight": state["weight"].clone(),
        "bias": state["bias"].clone() if "bias" in state else None,
        "normalize": bool(getattr(head, "normalize", True)),
    }


# ----------------------------------------------------------------------- images
def download_dataset(task: str, root: Path) -> Path:
    from huggingface_hub import snapshot_download

    spec = TASKS[task]
    target = root / "clip" / "datasets" / task
    splits = {"train", "test"} | {name for _, name in spec.split_names}
    patterns = [spec.files.format(split=split) + ("*" if task == "EMNIST" else "") for split in splits]
    if all(glob(str(target / pattern)) for pattern in patterns):
        return target  # already downloaded
    snapshot_download(spec.source, repo_type="dataset", local_dir=target, allow_patterns=patterns)
    return target


def load_raw(task: str, split: str, root: Path) -> tuple[list, np.ndarray]:
    """Encoded images (or arrays) and labels of the official ``train`` or ``test`` split."""
    spec = TASKS[task]
    split = dict(spec.split_names).get(split, split)
    pattern = str(root / "clip" / "datasets" / task / spec.files.format(split=split))
    if task == "FER2013":  # a pickled list of {"img_bytes", "labels"} with string labels
        with open(pattern, "rb") as handle:
            rows = pickle.load(handle)
        index = {name: i for i, name in enumerate(FER2013_LABELS)}
        return [row["img_bytes"] for row in rows], np.asarray(
            [index[row["labels"]] for row in rows], np.int64
        )
    if task == "EMNIST":  # raw idx files; like torchvision, the glyphs are not transposed
        with gzip.open(f"{pattern}-images-idx3-ubyte.gz", "rb") as handle:
            raw = handle.read()
        count, height, width = (int.from_bytes(raw[i : i + 4], "big") for i in (4, 8, 12))
        images = np.frombuffer(raw[16:], np.uint8).reshape(count, height, width)
        with gzip.open(f"{pattern}-labels-idx1-ubyte.gz", "rb") as handle:
            labels = np.frombuffer(handle.read()[8:], np.uint8).astype(np.int64)
        return list(images), labels
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"{task}/{split}: no files match {pattern}")
    table = pa.concat_tables([pq.read_table(file, columns=["image", "label"]) for file in files])
    images = [item["bytes"] if isinstance(item, dict) else item for item in table.column("image").to_pylist()]
    return images, np.asarray(table.column("label").to_pylist(), np.int64)


class _Images(torch.utils.data.Dataset):
    """Decode, resize (bicubic, shorter side 224) and center-crop to uint8 [3, 224, 224]."""

    def __init__(self, images: list, labels: np.ndarray) -> None:
        from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

        self.images, self.labels = images, labels
        self.transform = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224)])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        from PIL import Image

        item = self.images[index]
        image = Image.fromarray(item) if isinstance(item, np.ndarray) else Image.open(io.BytesIO(item))
        pixels = np.asarray(self.transform(image.convert("RGB")), dtype=np.uint8).copy()
        return torch.from_numpy(pixels).permute(2, 0, 1), self.labels[index]


def _cache(images: list, labels: np.ndarray, path: Path, workers: int, extra: dict | None = None) -> None:
    loader = torch.utils.data.DataLoader(_Images(images, labels), batch_size=256, num_workers=workers)
    batches = list(loader)
    payload = {
        "x": torch.cat([x for x, _ in batches]),
        "y": torch.cat([y for _, y in batches]),
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    print(f"images      {path}  {tuple(payload['x'].shape)}", flush=True)


def prepare_images(
    assets: ClipAssets, task: str, seeds, calibration_images: int, with_train: bool, workers: int
) -> None:
    wanted = {"test": assets.image_path(task, "test"), "val": assets.image_path(task, "val")}
    wanted.update({f"calib{seed}": assets.image_path(task, "calib", seed) for seed in seeds})
    if with_train:
        wanted["train"] = assets.image_path(task, "train")
    wanted = {
        name: path
        for name, path in wanted.items()
        if not path.exists()
        or (name.startswith("calib") and len(torch.load(path, weights_only=True)["y"]) < calibration_images)
    }
    if not wanted:
        return
    download_dataset(task, assets.root)
    if "test" in wanted:
        _cache(*load_raw(task, "test", assets.root), wanted.pop("test"), workers)
    if not wanted:
        return
    images, labels = load_raw(task, "train", assets.root)
    val_size = min(int(len(labels) * VAL_FRACTION), MAX_VAL)
    val_index = np.random.default_rng(SPLIT_SEED).permutation(len(labels))[:val_size]
    if "val" in wanted:
        _cache([images[i] for i in val_index], labels[val_index], wanted["val"], workers)
    for seed in seeds:
        if f"calib{seed}" in wanted:
            permutation = np.random.default_rng(seed).permutation(len(labels))
            index = permutation[val_size : val_size + calibration_images]
            _cache([images[i] for i in index], labels[index], wanted[f"calib{seed}"], workers)
    if "train" in wanted:  # joint training must exclude `val_index`
        _cache(images, labels, wanted["train"], workers, {"val_idx": torch.from_numpy(val_index.copy())})


# -------------------------------------------------------------------------- CLI
def run(args: argparse.Namespace) -> None:
    root = Path(args.data) if args.data else default_data_root()
    tasks = task_names(args.tasks) if args.only is None else tuple(args.only)
    for architecture in args.arch:
        if not args.skip_checkpoints:
            prepare_checkpoints(ClipAssets(architecture, root), tasks)
    assets = ClipAssets(args.arch[0], root)  # the image caches are shared by the backbones
    if not args.skip_images:
        for task in tasks:
            prepare_images(assets, task, args.seeds, args.calibration_images, args.train, args.workers)
    missing = [p for a in args.arch for p in ClipAssets(a, root).missing(tasks, seed=args.seeds[0])]
    print("ready" if not missing else f"still missing {len(missing)} files, e.g. {missing[0]}")


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "prepare-clip",
        help="download checkpoints and datasets, build image caches",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", nargs="+", default=["ViT-B-32"], choices=ARCHITECTURES)
    parser.add_argument("--tasks", type=int, default=8, choices=(8, 14, 20))
    parser.add_argument("--only", nargs="+", default=None, choices=ALL_TASKS, help="prepare these tasks only")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="calibration draws to build")
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=512,
        help="images per calibration draw; 512 = 256 statistics + 256 holdout, larger draws extend it",
    )
    parser.add_argument("--train", action="store_true", help="also cache the full training split")
    parser.add_argument("--data", default=None, help="data root (default: $IF_MERGE_DATA or ./data)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.set_defaults(run=run)
