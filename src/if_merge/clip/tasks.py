"""The 8/14/20-task CLIP merging benchmark.

The task order follows the TALL-masks / Iso-C benchmark: the 8-, 14- and 20-task
settings are the first 8, 14 and 20 entries of ``TASKS``.
"""

from __future__ import annotations

from dataclasses import dataclass

ARCHITECTURES = ("ViT-B-32", "ViT-B-16", "ViT-L-14")


@dataclass(frozen=True)
class Task:
    name: str
    classes: int
    #: Hugging Face dataset repository the images are read from
    source: str
    #: glob of the files of one split inside the repository, with a {split} placeholder
    files: str
    #: logical split -> name of the split in the repository, when they differ
    split_names: tuple[tuple[str, str], ...] = ()


TASKS: dict[str, Task] = {
    task.name: task
    for task in (
        Task("Cars", 196, "tanganke/stanford_cars", "data/{split}-*.parquet"),
        Task("DTD", 47, "tanganke/dtd", "data/{split}-*.parquet"),
        Task("EuroSAT", 10, "tanganke/eurosat", "data/{split}-*.parquet"),
        Task("GTSRB", 43, "tanganke/gtsrb", "data/{split}-*.parquet"),
        Task("MNIST", 10, "ylecun/mnist", "mnist/{split}-*.parquet"),
        Task("RESISC45", 45, "tanganke/resisc45", "data/{split}-*.parquet"),
        Task("SVHN", 10, "ufldl-stanford/svhn", "cropped_digits/{split}-*.parquet"),
        Task("SUN397", 397, "tanganke/sun397", "data/{split}-*.parquet"),
        Task("STL10", 10, "tanganke/stl10", "data/{split}-*.parquet"),
        Task("OxfordIIITPet", 37, "timm/oxford-iiit-pet", "data/{split}-*.parquet"),
        Task("Flowers102", 102, "dpdl-benchmark/oxford_flowers102", "data/{split}-*.parquet"),
        Task("CIFAR100", 100, "tanganke/cifar100", "data/{split}-*.parquet"),
        Task("PCAM", 2, "1aurent/PatchCamelyon", "data/{split}-*.parquet"),
        Task("FER2013", 7, "Jeneral/fer-2013", "{split}.pt"),
        Task("CIFAR10", 10, "tanganke/cifar10", "data/{split}-*.parquet"),
        Task(
            "Food101",
            101,
            "ethz/food101",
            "data/{split}-*.parquet",
            (("test", "validation"),),
        ),
        Task("FashionMNIST", 10, "zalando-datasets/fashion_mnist", "fashion_mnist/{split}-*.parquet"),
        Task("RenderedSST2", 2, "nateraw/rendered-sst2", "data/{split}-*.parquet"),
        Task("EMNIST", 10, "tanganke/emnist_digits", "raw/emnist-digits-{split}"),
        Task("KMNIST", 10, "tanganke/kmnist", "kmnist/{split}-*.parquet"),
    )
}
ALL_TASKS = tuple(TASKS)


def task_names(count: int) -> tuple[str, ...]:
    """The tasks of the 8-, 14- or 20-task setting."""
    if count not in (8, 14, 20):
        raise ValueError(f"The benchmark has 8, 14 or 20 tasks, not {count}.")
    return ALL_TASKS[:count]
