# Data

Everything is stored under one data root: `--data DIR`, or the `IF_MERGE_DATA` environment variable,
default `./data`. Nothing in it is tracked by git.

## CLIP

```bash
if-merge prepare-clip --arch ViT-B-32 --tasks 8                                  # 8-task ViT-B/32
if-merge prepare-clip --arch ViT-B-32 ViT-B-16 ViT-L-14 --tasks 20 --seeds 0 1 2  # everything in the paper
```

| What | Source | Stored as |
| :-- | :-- | :-- |
| Pretrained CLIP and the fine-tuned experts | [`nik-dim/tall_masks`](https://huggingface.co/nik-dim/tall_masks) | `clip/checkpoints/<arch>/{zeroshot,<Task>}.pt` |
| Zero-shot classification heads (frozen) | the same repository | `clip/heads/<arch>/<Task>.pt` |
| Images | one Hugging Face dataset per task, listed in [`src/if_merge/clip/tasks.py`](../src/if_merge/clip/tasks.py) | `clip/images/<Task>/{test,val,calib_seed<K>,train}.pt` |

The 8-, 14- and 20-task settings are the first 8, 14 and 20 tasks of the TALL-masks order:
Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SVHN, SUN397 · STL10, OxfordIIITPet, Flowers102, CIFAR100,
PCAM, FER2013 · CIFAR10, Food101, FashionMNIST, RenderedSST2, EMNIST, KMNIST.

Images are cached as `uint8` tensors after the geometric part of the CLIP preprocessing (bicubic resize
to 224, center crop); normalization happens on the GPU. The three backbones share the caches.

| Split | Content | Used for |
| :-- | :-- | :-- |
| `test` | the official test split | every reported number |
| `val` | 10% of the training split (at most 5,000 images), labeled | selecting the hyperparameters of baselines |
| `calib_seed<K>` | 512 training images outside `val`, unlabeled; draw `K` follows the permutation of `numpy.random.default_rng(K)` | IF-Merge: the first 256 images are $C_t$ (statistics), the next 256 are $H_t$ (step selection). Calibration data of the other methods |
| `train` | the full training split (`--train`) | the jointly trained reference only |

`--calibration-images N` builds longer draws; their first 512 images equal the default draw, so results
obtained with the default budget are unaffected.

Two tasks are kept exactly as the experts were trained, even where this looks wrong:

- **FER2013.** The labels follow the alphabetical order of the dataset (angry, disgust, fear, happy,
  neutral, sad, surprise), which differs from the class-name order behind the released zero-shot head.
  The experts were fine-tuned with this pairing.
- **EMNIST.** Glyphs are read from the raw files without the transposition some loaders apply, as in
  `torchvision.datasets.EMNIST`.

Disk, for all 20 tasks: checkpoints take about 9 GB per ViT-B backbone and 27 GB for ViT-L/14; the
image caches take about 46 GB (`test`, `val` and one calibration draw), plus about 150 GB for `train`.

## Flan-T5

See [`T5.md`](T5.md).
