# Reproducing the CLIP experiments

This guide lists, table by table, the commands behind the CLIP results of the paper and the
numbers to expect. Table numbers follow the paper; `S` marks the supplementary material.
The merging baselines are described in [`BASELINES.md`](BASELINES.md) and the Flan-T5
experiments in [`T5.md`](T5.md).

- [Conventions](#conventions)
- Main text: [Table 2](#table-2--clip-merging-performance) ·
  [Table 3](#table-3-s6-s7--if-merge-and-reforge-from-the-same-initialization) ·
  [Table 4](#table-4-s8-s9--distillation-from-the-same-initialization-and-images) ·
  [Table 6](#table-6-s20--two-sided-fitting) ·
  [Table 7](#table-7-s21-s23--single-round-multi-round-and-re-estimation)
- Supplementary: [S4](#table-s4--jointly-trained-reference-mtl) ·
  [S5](#table-s5--calibration-seeds) ·
  [S10](#table-s10--if-merge-from-other-merged-models) ·
  [S22](#table-s22--diagnostic-kl-of-the-fitting-and-iteration-controls) ·
  [S24, S25](#tables-s24-s25--step-size-selection-and-warm-start) ·
  [S27, S28, S29, S32](#tables-s27-s28-s29-s32--output-side-statistics) ·
  [S30](#table-s30--shrinkage-of-g) ·
  [S31](#table-s31--solver-budget-and-round-cap)
- [Endpoint analyses](#endpoint-analyses) (Fig. 3, Tables 8a, S34–S38, S40, S41)
- [Not covered by this release](#not-covered-by-this-release)

## Conventions

**Data.** Everything below assumes the data root built by `prepare-clip`
([`DATA.md`](DATA.md)):

```bash
if-merge prepare-clip --arch ViT-B-32 ViT-B-16 ViT-L-14 --tasks 20 --seeds 0 1 2
```

**Settings.** `--arch` is `ViT-B-32`, `ViT-B-16` or `ViT-L-14`; `--tasks` is 8, 14 or 20.
"B/32·8" below means ViT-B/32 with 8 tasks.

**Recipe.** IF-Merge always runs [`configs/if_merge/clip.yaml`](../configs/if_merge/clip.yaml):
256 unlabeled calibration images per task for the statistics (C_t), 256 for the holdout KL
(H_t), at most four rounds, conjugate gradient capped at 100 iterations, one global step
size per round reselected per Transformer block. An ablation changes one field with
`--set section.field=value`.

**Initializations.** `--init` takes a baseline id (`wa`, `ta`, `ties`, `tsv_m`, `wudi`,
`iso_c`, `iso_cts`, `regmean`, `regmeanpp`, `fisher`, `mats`, `adamerging`) or the path of a
checkpoint. A baseline is read from an earlier `if-merge baseline` run if there is one, and
otherwise rebuilt with its validation-selected hyperparameters
(`configs/baselines/clip_selected.yaml`).

**Seeds.** `--seed K` selects calibration draw `K`, that is, C_t and H_t. The experts, the
initialization and the test set do not change. Multi-seed results are means ± sample
standard deviations over seeds 0–2, computed as in the paper from per-seed accuracies
rounded to two decimals.

**Outputs.** A run writes `result.json` and `merged.pt` to
`runs/clip/<arch>-<T>tasks/<command>/<name>/`; `--name` overrides the name, `--no-save`
skips `merged.pt`, and a finished run is skipped unless `--force` is given.
`result.json` holds the test accuracy of the initialization (`before`) and of the selected
model (`after`), per task and averaged, and for IF-Merge the holdout KL, the step sizes and
the time spent measuring the statistics in every round (`rounds`) and the selected round.
`if-merge clip --save-rounds` also writes the merged model after every round (`round<r>.pt`).

**Diagnostic KL.** `if-merge clip --diagnostic-kl` adds the expert–merged KL on images that
take no part in the method: the first 256 images of the labeled `val` split of every task
(all 102 for Flowers102), for the initialization (`diagnostic_kl.before`) and for the
selected model (`diagnostic_kl.after`). The holdout KL, in contrast, is the quantity the
step sizes and the round are selected by.

**Reference numbers.** [`results/clip_if_merge.json`](../results/clip_if_merge.json) lists
every IF-Merge run with the default recipe: per backbone, task set, initialization and
seed, the accuracy before and after, the selected round and the holdout KL of every round.
The baselines are in [`results/clip_baselines.json`](../results/clip_baselines.json).
All accuracies are task-averaged test accuracies in percent. Runs are deterministic on a
given machine; other hardware or library versions can change the last digits.

**Cost of one IF-Merge run** (default recipe; measured on 48 GB GPUs shared between jobs,
a dedicated GPU is about twice as fast):

| Backbone | 8 tasks | 14 tasks | 20 tasks | Peak GPU memory |
| :-- | --: | --: | --: | --: |
| ViT-B/32 | 20 min | 35 min | 25–40 min | 4 GB |
| ViT-B/16 | 40 min | 65–80 min | 100–115 min | 10 GB |
| ViT-L/14 | 2.4 h | 4–6 h | 6 h | 28 GB |

---

## Table 2 — CLIP merging performance

IF-Merge from Weight Averaging, Task Arithmetic and Iso-CTS on three backbones and three
task sets, next to the merging baselines and the jointly trained reference.

```bash
for arch in ViT-B-32 ViT-B-16 ViT-L-14; do
  for tasks in 8 14 20; do
    for init in wa ta iso_cts; do
      for seed in 0 1 2; do
        if-merge clip --arch $arch --tasks $tasks --init $init --seed $seed
      done
    done
  done
done
```

Expected, mean ± SD over the three seeds:

| IF-Merge from | B/32·8 | B/32·14 | B/32·20 | B/16·8 | B/16·14 | B/16·20 | L/14·8 | L/14·14 | L/14·20 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| `wa` | 87.00 ± 0.04 | 84.46 ± 0.09 | 84.51 ± 0.08 | 89.38 ± 0.13 | 86.97 ± 0.00 | 86.40 ± 0.10 | 92.04 ± 0.03 | 90.13 ± 0.02 | 90.02 ± 0.02 |
| `ta` | 86.96 ± 0.01 | 84.44 ± 0.07 | 84.53 ± 0.11 | 89.38 ± 0.10 | 86.96 ± 0.11 | 86.45 ± 0.06 | 92.06 ± 0.03 | 90.08 ± 0.04 | 90.02 ± 0.10 |
| `iso_cts` | 88.81 ± 0.08 | 86.02 ± 0.09 | 85.62 ± 0.09 | 91.40 ± 0.06 | 88.75 ± 0.14 | 88.29 ± 0.07 | 93.75 ± 0.02 | 91.87 ± 0.03 | 91.92 ± 0.06 |

The per-seed values are in [Table S5](#table-s5--calibration-seeds). The baseline rows
are produced by `if-merge baseline` ([`BASELINES.md`](BASELINES.md)) and the MTL row by
`if-merge joint` ([Table S4](#table-s4--jointly-trained-reference-mtl)). The zero-shot row
evaluates the pretrained checkpoint (48.11 on B/32·8):

```bash
if-merge evaluate --arch ViT-B-32 --tasks 8 --checkpoint data/clip/checkpoints/ViT-B-32/zeroshot.pt
```

## Table 3, S6, S7 — IF-Merge and ReForge from the same initialization

Six initializations refined by IF-Merge; the same merged models are refined by ReForge
(Table 3, S7) and by LiNeS and SVC (Table S6).

```bash
for tasks in 8 14 20; do
  for init in wa ta ties tsv_m wudi iso_cts; do
    for seed in 0 1 2; do
      if-merge clip --arch ViT-B-32 --tasks $tasks --init $init --seed $seed
      if-merge baseline --arch ViT-B-32 --tasks $tasks --method reforge --init $init --seed $seed
    done
  done
done
```

Table S7 adds ViT-B/16 with 8 tasks (same loop with `--arch ViT-B-16 --tasks 8`). Table S6
uses seed 0 and adds `--method lines` and `--method svc`. Expected IF-Merge accuracy,
mean ± SD over seeds 0–2 (the ReForge, LiNeS and SVC numbers are in
[`BASELINES.md`](BASELINES.md)):

| Initialization | B/32·8 | B/32·14 | B/32·20 | B/16·8 |
| :-- | --: | --: | --: | --: |
| `wa` | 87.00 ± 0.04 | 84.46 ± 0.09 | 84.51 ± 0.08 | 89.38 ± 0.13 |
| `ta` | 86.96 ± 0.01 | 84.44 ± 0.07 | 84.53 ± 0.11 | 89.38 ± 0.10 |
| `ties` | 87.02 ± 0.13 | 84.51 ± 0.08 | 84.44 ± 0.10 | 89.40 ± 0.07 |
| `tsv_m` | 87.83 ± 0.15 | 85.07 ± 0.03 | 84.67 ± 0.23 | 90.25 ± 0.07 |
| `wudi` | 87.97 ± 0.14 | 85.05 ± 0.06 | 84.65 ± 0.23 | 90.57 ± 0.10 |
| `iso_cts` | 88.81 ± 0.08 | 86.02 ± 0.09 | 85.62 ± 0.09 | 91.40 ± 0.06 |

On ViT-B/32 with 20 tasks the selected Task Arithmetic scale is 1/20, so `ta` and `wa` are
the same model.

## Table 4, S8, S9 — Distillation from the same initialization and images

Teacher–student distillation (KD) starts from the same merged model as IF-Merge and sees
the same unlabeled images: it trains on C_t and selects its learning rate and number of
steps by the holdout KL on H_t (protocol in
[`src/if_merge/clip/distill.py`](../src/if_merge/clip/distill.py)). The budget `--images N`
is |C_t| + |H_t| per task, for `if-merge distill` and for `if-merge clip` alike. Up to 512
images the draw is halved (C_t first, H_t second); larger budgets keep the 256 holdout
images of the default budget and add all further images to C_t. Budgets above 512 need
longer calibration draws, whose first 512 images are the default draw:

```bash
if-merge prepare-clip --arch ViT-B-32 ViT-B-16 --tasks 8 --seeds 0 1 2 --calibration-images 2048
```

```bash
for arch in ViT-B-32 ViT-B-16; do
  for init in wa iso_cts; do
    for seed in 0 1 2; do
      for images in 128 512 2048; do
        if-merge distill --arch $arch --tasks 8 --init $init --seed $seed --images $images
        if-merge clip    --arch $arch --tasks 8 --init $init --seed $seed --images $images
      done
    done
  done
done
```

Expected KD accuracy (Table 4 and S9; seed 0, then mean ± SD over seeds 0–2), with the
learning rate and step selected at seed 0:

| Backbone | Initialization | Images | Seed 0 | Selected (seed 0) | Mean ± SD |
| :-- | :-- | --: | --: | :-- | --: |
| B/32 | `wa` | 128 | 75.83 | 3e-6, step 120 | 75.49 ± 0.48 |
| B/32 | `wa` | 512 | 80.30 | 1e-5, step 448 | 80.27 ± 0.07 |
| B/32 | `wa` | 2,048 | 85.92 | 3e-6, step 3,000 | 86.14 ± 0.21 |
| B/32 | `iso_cts` | 128 | 87.36 | 1e-6, step 2,776 | 87.20 ± 0.18 |
| B/32 | `iso_cts` | 512 | 88.49 | 1e-6, step 6,432 | seeds 1, 2: 88.55, 88.37 |
| B/32 | `iso_cts` | 2,048 | 89.50 | 1e-6, step 6,600 | seeds 1, 2: 89.53, 89.67 |
| B/16 | `wa` | 128 | 78.84 | 3e-6, step 120 | 78.87 ± 0.07 |
| B/16 | `wa` | 512 | 83.37 | 1e-5, step 288 | 83.31 ± 0.10 |
| B/16 | `wa` | 2,048 | 88.00 | 3e-6, step 1,900 | 88.00 ± 0.00 |
| B/16 | `iso_cts` | 128 | 90.40 | 3e-7, step 11,944 | 90.41 ± 0.09 |
| B/16 | `iso_cts` | 512 | 91.18 | 1e-6, step 4,864 | 91.14 ± 0.10 |
| B/16 | `iso_cts` | 2,048 | 91.94 | 1e-6, step 7,400 | 91.95 ± 0.01 |

A selected step above 6,000 means that the learning rate was still improving at the step
cap and was trained again with twice the cap.

Expected IF-Merge accuracy with the same images, mean ± SD over seeds 0–2 (Table 4, S9):

| Backbone | Initialization | 128 | 512 | 2,048 |
| :-- | :-- | --: | --: | --: |
| B/32 | `wa` | 85.05 ± 0.15 | 87.00 ± 0.04 | 88.14 ± 0.09 |
| B/32 | `iso_cts` | 88.17 ± 0.10 | 88.81 ± 0.08 | 89.23 ± 0.07 |
| B/16 | `wa` | 87.56 ± 0.13 | 89.38 ± 0.13 | 90.56 ± 0.12 |
| B/16 | `iso_cts` | 91.05 ± 0.08 | 91.40 ± 0.06 | 91.56 ± 0.16 |

and at seed 0 over all budgets of Table S8a (`--images 64` … `--images 3072`; the last
needs `--calibration-images 3072`):

| Backbone | Initialization | 64 | 128 | 256 | 512 | 1,024 | 2,048 | 3,072 |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| B/32 | `wa` | 83.60 | 84.96 | 86.36 | 87.02 | 87.60 | 88.14 | 88.38 |
| B/32 | `iso_cts` | 87.92 | 88.06 | 88.55 | 88.72 | 88.98 | 89.16 | 89.16 |
| B/16 | `wa` | – | 87.46 | – | 89.24 | – | 90.44 | – |
| B/16 | `iso_cts` | – | 91.11 | – | 91.33 | – | 91.52 | – |

**Table S8a, pretrained weights.** The pretrained model is passed as a checkpoint:

```bash
if-merge distill --arch ViT-B-32 --tasks 8 --init data/clip/checkpoints/ViT-B-32/zeroshot.pt --images 2048
if-merge clip    --arch ViT-B-32 --tasks 8 --init data/clip/checkpoints/ViT-B-32/zeroshot.pt --images 2048
if-merge distill --arch ViT-L-14 --tasks 8 --init data/clip/checkpoints/ViT-L-14/zeroshot.pt --images 2048 \
  --tasks-per-forward 1
```

| Method, from pretrained weights | 128 | 512 | 1,024 | 2,048 | 3,072 |
| :-- | --: | --: | --: | --: | --: |
| KD, B/32 | 65.49 | 76.33 | 81.94 | 84.96 | 86.51 |
| KD, B/16 | 70.89 | 79.95 | – | 87.49 | – |
| KD, L/14 | – | – | – | 89.91 | – |
| IF-Merge, B/32 | – | 87.03 | – | 88.12 | – |
| IF-Merge, B/16 | – | 89.27 | – | – | – |

**Table S8b, 512 images, seed 0.** `if-merge distill` and `if-merge clip` with `--init`
in `wa ta regmean tsv_m wudi iso_c`; each cell is IF-Merge / KD:

| Backbone | Initialization | Accuracy | Holdout KL |
| :-- | :-- | --: | --: |
| B/32 | `wa` | 87.02 / 80.30 | 0.206 / 0.559 |
| B/32 | `ta` | 86.95 / 83.04 | 0.210 / 0.408 |
| B/32 | `regmean` | 86.91 / 86.15 | 0.208 / 0.235 |
| B/32 | `tsv_m` | 87.80 / 87.90 | 0.174 / 0.174 |
| B/32 | `wudi` | 88.01 / 88.37 | 0.147 / 0.140 |
| B/32 | `iso_c` | 88.26 / 88.35 | 0.112 / 0.101 |
| B/16 | `iso_c` | 90.99 / 91.10 | 0.064 / 0.062 |

**Cost of KD.** One search trains three to five learning rates. Measured on shared GPUs:
ViT-B/32 from `wa` with 512 images 37 min and 7 GB; from `iso_cts` with 512 images 2.7 h
(the strong initializations select small learning rates that train for thousands of
steps); ViT-B/16 1–6 h and 20 GB; ViT-L/14 with one task per forward pass 11 h and 38 GB.

## Table 6, S20 — Two-sided fitting

Replacing the activation statistics A, the output-side statistics G, or both by the
identity. With both replaced, every layer is fitted to the plain mean of the expert weights.

```bash
for tasks in 8 14 20; do
  for init in wa ta iso_cts iso_c; do
    for seed in 0 1 2; do
      run="if-merge clip --arch ViT-B-32 --tasks $tasks --init $init --seed $seed"
      $run --set a.estimator=identity --set g.estimator=identity   # (I, I)
      $run --set g.estimator=identity                              # (A, I)
      $run --set a.estimator=identity                              # (I, G)
      $run                                                         # (A, G)
    done
  done
done
```

Expected, mean ± SD over seeds 0–2 (Table S20 lists the seeds):

| Tasks | Initialization | (I, I) | (A, I) | (I, G) | (A, G) |
| --: | :-- | --: | --: | --: | --: |
| 8 | `wa` | 65.50 ± 0.00 | 85.45 ± 0.11 | 82.46 ± 0.32 | 87.00 ± 0.04 |
| 8 | `ta` | 73.04 ± 0.07 | 85.48 ± 0.09 | 82.47 ± 0.22 | 86.96 ± 0.01 |
| 8 | `iso_cts` | 85.19 ± 0.06 | 88.13 ± 0.14 | 87.27 ± 0.07 | 88.81 ± 0.08 |
| 8 | `iso_c` | 84.32 ± 0.01 | 87.47 ± 0.07 | 86.48 ± 0.06 | 88.29 ± 0.13 |
| 14 | `wa` | 64.22 ± 0.00 | 81.01 ± 0.09 | 79.29 ± 0.13 | 84.46 ± 0.09 |
| 14 | `ta` | 65.88 ± 0.07 | 80.99 ± 0.14 | 79.33 ± 0.16 | 84.44 ± 0.07 |
| 14 | `iso_cts` | 81.51 ± 0.07 | 84.76 ± 0.05 | 84.00 ± 0.01 | 86.02 ± 0.09 |
| 14 | `iso_c` | 80.17 ± 0.00 | 83.69 ± 0.10 | 82.95 ± 0.02 | 85.46 ± 0.06 |
| 20 | `wa` | 60.80 ± 0.01 | 78.24 ± 0.15 | 76.76 ± 0.07 | 84.51 ± 0.08 |
| 20 | `ta` | 60.81 ± 0.00 | 78.17 ± 0.20 | 76.74 ± 0.08 | 84.53 ± 0.11 |
| 20 | `iso_cts` | 78.66 ± 0.00 | 82.15 ± 0.01 | 81.57 ± 0.05 | 85.62 ± 0.09 |
| 20 | `iso_c` | 76.42 ± 0.01 | 80.80 ± 0.03 | 80.15 ± 0.04 | 85.17 ± 0.10 |

The two additional seed-0 rows of Table S20 use `--init tsv_m` on B/32·8
(84.22, 87.27, 85.75, 87.80) and `--arch ViT-B-16 --init iso_c` on 8 tasks
(88.93, 90.71, 89.70, 90.99).

## Table 7, S21, S23 — Single round, multi-round and re-estimation

One round with 100 or 400 conjugate-gradient iterations, four rounds with the statistics of
the first round, and four rounds with A, G or both measured again after every update.

```bash
for tasks in 8 14 20; do
  for init in wa ta iso_cts iso_c; do
    for seed in 0 1 2; do
      run="if-merge clip --arch ViT-B-32 --tasks $tasks --init $init --seed $seed"
      $run --set iteration.rounds=1                                    # single round, CG100
      $run --set iteration.rounds=1 --set solver.max_iterations=400    # single round, CG400
      $run --set iteration.reestimate=none                             # multi-round, fixed statistics
      $run --set iteration.reestimate=a                                # only A re-estimated
      $run --set iteration.reestimate=g                                # only G re-estimated
      $run                                                             # joint re-estimation
    done
  done
done
```

Table 7 and S21 report the single-round runs for `ta` and `iso_c` only. Expected, mean ± SD
over seeds 0–2:

| Tasks | Initialization | Single, CG100 | Single, CG400 | Fixed statistics | Joint re-estimation |
| --: | :-- | --: | --: | --: | --: |
| 8 | `ta` | 85.78 ± 0.18 | 86.20 ± 0.18 | 86.19 ± 0.13 | 86.96 ± 0.01 |
| 8 | `iso_c` | 88.20 ± 0.09 | 87.97 ± 0.08 | 88.24 ± 0.06 | 88.29 ± 0.13 |
| 14 | `ta` | 82.28 ± 0.12 | 83.08 ± 0.13 | 83.02 ± 0.13 | 84.44 ± 0.07 |
| 14 | `iso_c` | 85.04 ± 0.05 | 84.81 ± 0.18 | 85.17 ± 0.08 | 85.46 ± 0.06 |
| 20 | `ta` | 79.79 ± 0.20 | 81.07 ± 0.22 | 80.88 ± 0.21 | 84.53 ± 0.11 |
| 20 | `iso_c` | 83.88 ± 0.08 | 83.76 ± 0.12 | 83.94 ± 0.04 | 85.17 ± 0.10 |

Table S23, accuracy with fixed statistics and the paired gain of each re-estimation
strategy over it (same seed), mean ± SD:

| Tasks | Initialization | Fixed | Only A | Only G | Joint |
| --: | :-- | --: | --: | --: | --: |
| 8 | `wa` | 86.30 ± 0.09 | +0.54 ± 0.07 | +0.43 ± 0.09 | +0.70 ± 0.05 |
| 8 | `ta` | 86.19 ± 0.13 | +0.79 ± 0.18 | +0.40 ± 0.10 | +0.77 ± 0.14 |
| 8 | `iso_cts` | 88.71 ± 0.08 | +0.12 ± 0.03 | +0.05 ± 0.07 | +0.09 ± 0.01 |
| 8 | `iso_c` | 88.24 ± 0.06 | +0.03 ± 0.07 | +0.05 ± 0.06 | +0.05 ± 0.09 |
| 14 | `wa` | 82.94 ± 0.11 | +0.65 ± 0.08 | +1.14 ± 0.18 | +1.52 ± 0.02 |
| 14 | `ta` | 83.02 ± 0.13 | +0.71 ± 0.11 | +0.97 ± 0.17 | +1.42 ± 0.09 |
| 14 | `iso_cts` | 85.79 ± 0.08 | +0.16 ± 0.03 | +0.07 ± 0.02 | +0.23 ± 0.02 |
| 14 | `iso_c` | 85.17 ± 0.08 | +0.22 ± 0.01 | +0.12 ± 0.02 | +0.29 ± 0.03 |
| 20 | `wa` | 80.89 ± 0.22 | +1.43 ± 0.27 | +2.83 ± 0.29 | +3.63 ± 0.16 |
| 20 | `ta` | 80.88 ± 0.21 | +1.33 ± 0.12 | +2.86 ± 0.28 | +3.64 ± 0.13 |
| 20 | `iso_cts` | 84.89 ± 0.01 | +0.32 ± 0.03 | +0.41 ± 0.06 | +0.72 ± 0.10 |
| 20 | `iso_c` | 83.94 ± 0.04 | +0.62 ± 0.07 | +0.74 ± 0.06 | +1.23 ± 0.09 |

A single round takes 4–12 min on ViT-B/32 with 8 tasks, 16–18 min with 400 iterations.

---

## Table S4 — Jointly trained reference (MTL)

The MTL row of Table 2: the pretrained visual encoder fine-tuned on the labeled training
splits of all tasks for 3,000 steps (recipe in
[`src/if_merge/clip/joint.py`](../src/if_merge/clip/joint.py)), seed 0, evaluated with the
same heads and test sets as the merged models.

```bash
if-merge prepare-clip --arch ViT-B-32 --tasks 14 --train     # caches the full training splits
if-merge joint --arch ViT-B-32 --tasks 14
```

| | B/32·8 | B/32·14 | B/32·20 | B/16·8 | B/16·14 | B/16·20 | L/14·8 | L/14·14 | L/14·20 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| MTL | 88.53† | 88.28 | 88.35 | 91.94 | 90.40 | 90.58 | 92.29† | 92.59 | 93.13 |

† These two cells are not trained here. They evaluate the jointly trained 8-task
checkpoints released with FusionBench,
[`tanganke/clip-vit-base-patch32_TA8_mtl`](https://huggingface.co/tanganke/clip-vit-base-patch32_TA8_mtl)
and
[`tanganke/clip-vit-large-patch14_TA8_mtl`](https://huggingface.co/tanganke/clip-vit-large-patch14_TA8_mtl),
with the heads and test sets of this benchmark. The released weights have no output
projection; the pretrained one is used.

```bash
pip install huggingface-hub safetensors
if-merge joint --arch ViT-B-32 --tasks 8 --released
if-merge joint --arch ViT-L-14 --tasks 8 --released
```

The other seven cells are trained by `if-merge joint`. Training takes 7–14 min and 5 GB on
ViT-B/32, 31–41 min and 10 GB on ViT-B/16, and 2.4–2.5 h and 21 GB on ViT-L/14; the
training caches of the 20-task benchmark take about 150 GB of disk and are memory-mapped.
`if-merge joint` evaluates the model at the end of training, in the same process;
evaluating the saved `model.pt` again with `if-merge evaluate` can differ on a few test
images (88.29 instead of 88.28 on B/32·14), because training enables TF32 matrix products.

## Table S5 — Calibration seeds

The runs of [Table 2](#table-2--clip-merging-performance), per seed:

| Initialization | Seed | B/32·8 | B/32·14 | B/32·20 | B/16·8 | B/16·14 | B/16·20 | L/14·8 | L/14·14 | L/14·20 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| `wa` | 0 | 87.02 | 84.56 | 84.55 | 89.24 | 86.97 | 86.51 | 92.02 | 90.13 | 90.01 |
| `wa` | 1 | 87.02 | 84.39 | 84.57 | 89.50 | 86.98 | 86.30 | 92.08 | 90.11 | 90.00 |
| `wa` | 2 | 86.95 | 84.42 | 84.43 | 89.39 | 86.97 | 86.39 | 92.03 | 90.14 | 90.04 |
| `ta` | 0 | 86.95 | 84.51 | 84.56 | 89.28 | 86.90 | 86.46 | 92.04 | 90.07 | 90.02 |
| `ta` | 1 | 86.96 | 84.37 | 84.62 | 89.41 | 87.09 | 86.39 | 92.05 | 90.04 | 90.11 |
| `ta` | 2 | 86.97 | 84.44 | 84.40 | 89.47 | 86.88 | 86.50 | 92.09 | 90.12 | 89.92 |
| `iso_cts` | 0 | 88.72 | 86.11 | 85.54 | 91.33 | 88.83 | 88.25 | 93.73 | 91.87 | 91.98 |
| `iso_cts` | 1 | 88.89 | 86.02 | 85.58 | 91.44 | 88.83 | 88.37 | 93.77 | 91.84 | 91.86 |
| `iso_cts` | 2 | 88.81 | 85.93 | 85.72 | 91.43 | 88.59 | 88.25 | 93.75 | 91.89 | 91.92 |

## Table S10 — IF-Merge from other merged models

```bash
for init in tsv_m wudi iso_c fisher regmean mats adamerging regmeanpp; do
  if-merge clip --arch ViT-B-32 --tasks 8 --init $init
done
```

and the same with `--tasks 20` and with `--arch ViT-B-16 --tasks 8`. Expected endpoint
accuracy, seed 0:

| Initialization | B/32·8 | B/32·20 | B/16·8 |
| :-- | --: | --: | --: |
| `tsv_m` | 87.80 | 84.88 | 90.25 |
| `wudi` | 88.01 | 84.65 | 90.52 |
| `iso_c` | 88.26 | 85.15 | 90.99 |
| `fisher` | 87.03 | 84.58 | 89.04 |
| `regmean` | 86.91 | 84.65 | 89.53 |
| `mats` | 87.01 | 84.71 | 89.57 |
| `adamerging` | 87.15 | 84.77 | 89.19 |
| `regmeanpp` | 87.07 | 84.69 | 89.27 |

The accuracy of an initialization in `result.json` (`before`) can differ by a few test
images from the number `if-merge baseline` reports for the same merged model: the two
evaluate the encoder in different layouts under float16 autocast
([`BASELINES.md`](BASELINES.md)). The gap is largest for RegMean, whose weights are large
(B/32·8: 83.51 here, 83.59 as a baseline, 83.61 for both in float32).

## Table S22 — Diagnostic KL of the fitting and iteration controls

The runs of [Table 6](#table-6-s20--two-sided-fitting) and
[Table 7](#table-7-s21-s23--single-round-multi-round-and-re-estimation) with
`--diagnostic-kl`; the table reports `diagnostic_kl.after.average`. For example, the
fixed-statistics run of the first row:

```bash
if-merge clip --arch ViT-B-32 --tasks 20 --init ta --seed 1 --diagnostic-kl --set iteration.reestimate=none
```

| Setting | Seed | (I, I) | (A, I) | (I, G) | (A, G) | Fixed | Only A | Only G | Single, CG100 | Single, CG400 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| B/32·20 `ta` | 1 | 1.1342 | 0.5484 | 0.6610 | 0.3238 | 0.4618 | 0.4063 | 0.3509 | 0.5155 | 0.4536 |
| B/32·20 `ta` | 2 | 1.1342 | 0.5445 | 0.6599 | 0.3222 | 0.4616 | 0.4065 | 0.3510 | 0.5163 | 0.4566 |
| B/32·8 `iso_c` | 1 | 0.2688 | 0.1562 | 0.2026 | 0.1130 | 0.1224 | 0.1187 | 0.1206 | 0.1241 | 0.1325 |
| B/32·8 `iso_c` | 2 | 0.2717 | 0.1540 | 0.2109 | 0.1220 | 0.1255 | 0.1243 | 0.1246 | 0.1265 | 0.1379 |

## Tables S24, S25 — Step-size selection and warm start

Table S24 compares a fixed step size in every round, one step size per round selected
for the full model, and the default (selected for the full model, then reselected per
Transformer block). `selection.forced_steps` lists the step size imposed in round 1, 2, …;
a forced round takes that step in every layer without consulting the holdout KL, and the
round is still selected by holdout KL.

```bash
run="if-merge clip --arch ViT-B-32 --tasks 8 --init ta --diagnostic-kl"
for step in 0.125 0.25 0.5 1.0; do
  $run --set "selection.forced_steps=[$step,$step,$step,$step]"   # fixed step size
done
$run --set selection.scope=global                                  # global only
$run                                                               # global + per block
```

Each cell is the accuracy / the holdout KL / the diagnostic KL of the selected round and,
in parentheses, the selected round (seed 0). The first row is the initialization.

| Update rule | B/32·8 `ta` | B/32·8 `iso_c` | B/32·20 `ta` | B/16·8 `iso_c` |
| :-- | --: | --: | --: | --: |
| Initialization | 69.98 / 0.9816 / 0.9620 | 83.82 / 0.2630 / 0.2802 | 60.81 / 1.1625 / 1.1342 | 88.49 / 0.1674 / 0.1860 |
| Fixed step 1/8 | 82.60 / 0.4508 / 0.4369 (4) | 87.82 / 0.1354 / 0.1392 (4) | 75.74 / 0.6918 / 0.6753 (4) | 90.62 / 0.0861 / 0.1092 (4) |
| Fixed step 1/4 | 85.44 / 0.3017 / 0.2934 (4) | 88.06 / 0.1304 / 0.1303 (3) | 80.67 / 0.4938 / 0.4840 (4) | 90.70 / 0.0813 / 0.1023 (3) |
| Fixed step 1/2 | 86.92 / 0.2239 / 0.2222 (4) | 88.05 / 0.1266 / 0.1296 (1) | 83.73 / 0.3701 / 0.3624 (4) | 90.65 / 0.0825 / 0.1021 (1) |
| Fixed step 1 | 86.93 / 0.2122 / 0.2094 (4) | 87.14 / 0.1738 / 0.1714 (1) | 83.54 / 0.3706 / 0.3723 (4) | 89.64 / 0.1284 / 0.1275 (1) |
| Global only | 87.16 / 0.2102 / 0.2084 (4) | 88.09 / 0.1266 / 0.1279 (2) | 84.51 / 0.3416 / 0.3380 (4) | 90.66 / 0.0818 / 0.1009 (2) |
| Global + per block | 86.95 / 0.2099 / 0.2119 (4) | 88.26 / 0.1124 / 0.1204 (4) | 84.56 / 0.3382 / 0.3343 (4) | 90.99 / 0.0644 / 0.0862 (4) |

The accepted step sizes of Table S24(b) are the `global_step` and `block_steps` entries of
`rounds` in `result.json`.

Table S25 varies the initial guess of the conjugate-gradient solver (the current weights,
or the mean of the expert weights) and forces a full step in the first round only, on
B/32·8 from `iso_c`, seed 0:

```bash
run="if-merge clip --arch ViT-B-32 --tasks 8 --init iso_c"
$run --set solver.warm_start=expert_mean                                           # E2-a
$run --set 'selection.forced_steps=[1.0]'                                          # E2-b
$run --set 'selection.forced_steps=[1.0]' --set solver.warm_start=expert_mean      # E2-c
```

| Run | Initial guess | First round | Holdout KL | Accuracy | Selected round |
| :-- | :-- | :-- | --: | --: | --: |
| Default | current weights | selected | 0.1124 | 88.26 | 4 |
| E2-a | expert mean | selected | 0.1320 | 87.92 | 4 |
| E2-b | current weights | full step | 0.1663 | 87.43 | 3 |
| E2-c | expert mean | full step | 0.2252 | 86.65 | 4 |

## Tables S27, S28, S29, S32 — Output-side statistics

Alternative constructions of G and controls on which task a G belongs to; everything else
is the default recipe.

```bash
run="if-merge clip --arch ViT-B-32 --tasks 8 --init ta --diagnostic-kl"
for estimator in normalized_kl identity kl exact_fisher mc_fisher empirical_fisher output_activation; do
  $run --set g.estimator=$estimator
done
$run --set g.permute_tasks=true     # task t is fitted with the G of task t + 1
$run --set g.heterogeneity=0.0      # every task is fitted with the mean G of the layer
$run --set g.heterogeneity=0.5      # half of each task's deviation from the mean G is kept
```

| Switch | Row of Table S27 / column of Table S28 |
| :-- | :-- |
| `g.estimator=normalized_kl` | Normalized residual G (default); "Original task G" |
| `g.estimator=identity` | G = I; "A only" |
| `g.estimator=kl` | Without χ² normalization |
| `g.estimator=exact_fisher` | Exact predictive Fisher (one backward pass per class) |
| `g.estimator=mc_fisher` | Monte Carlo Fisher (four sampled labels per image, `g.mc_samples`) |
| `g.estimator=empirical_fisher` | Empirical Fisher (uses the labels of the calibration images) |
| `g.estimator=output_activation` | Layer-output activation second moment (a forward statistic) |
| `g.permute_tasks=true` | Cross-task permuted G; "Permuted G" |
| `g.heterogeneity=0.0`, `0.5` | "Shared G", "η = 0.5" |

Table S27, accuracy / holdout KL / diagnostic KL (selected round), seed 0:

| Output factor | B/32·8 `ta` | B/32·8 `iso_c` | B/32·20 `ta` | B/16·8 `iso_c` |
| :-- | --: | --: | --: | --: |
| `normalized_kl` | 86.95 / 0.2099 / 0.2119 (4) | 88.26 / 0.1124 / 0.1204 (4) | 84.56 / 0.3382 / 0.3343 (4) | 90.99 / 0.0644 / 0.0862 (4) |
| `identity` | 85.58 / 0.2736 / 0.2785 (4) | 87.55 / 0.1534 / 0.1549 (4) | 78.40 / 0.5480 / 0.5378 (4) | 90.71 / 0.0876 / 0.1014 (3) |
| `kl` | 86.63 / 0.2241 / 0.2270 (4) | 87.96 / 0.1232 / 0.1302 (4) | 83.51 / 0.3606 / 0.3688 (4) | 90.90 / 0.0721 / 0.0924 (3) |
| `exact_fisher` | 87.21 / 0.1887 / 0.1924 (4) | 88.37 / 0.1039 / 0.1130 (4) | 84.92 / 0.3104 / 0.3121 (4) | 91.07 / 0.0607 / 0.0812 (3) |
| `mc_fisher` | 86.90 / 0.2046 / 0.2115 (4) | 88.34 / 0.1074 / 0.1203 (3) | 84.30 / 0.3340 / 0.3330 (4) | 90.98 / 0.0624 / 0.0873 (3) |
| `empirical_fisher` | 86.55 / 0.2242 / 0.2243 (4) | 87.95 / 0.1238 / 0.1305 (4) | 83.36 / 0.3625 / 0.3739 (4) | 90.86 / 0.0704 / 0.0900 (2) |
| `output_activation` | 77.69 / 0.5940 / 0.6013 (4) | 85.57 / 0.1973 / 0.2116 (4) | 67.68 / 0.9265 / 0.9025 (4) | 89.69 / 0.1139 / 0.1434 (4) |
| permuted across tasks | 81.88 / 0.4052 / 0.4135 (4) | 86.55 / 0.1753 / 0.1957 (4) | 72.96 / 0.7666 / 0.7486 (4) | 90.23 / 0.1025 / 0.1165 (4) |

Table S28, accuracy at seed 0. The diagnostic KL reduction is `diagnostic_kl.after` of the
`identity` run minus that of the default run:

| Setting | Diagnostic KL reduction | A only | Shared G | η = 0.5 | Original task G | Permuted G |
| :-- | --: | --: | --: | --: | --: | --: |
| B/32·8 `ta` | +0.0666 | 85.58 | 85.47 | 86.54 | 86.95 | 81.88 |
| B/32·8 `iso_c` | +0.0345 | 87.55 | 87.59 | 88.08 | 88.26 | 86.55 |
| B/32·20 `ta` | +0.2035 | 78.40 | 78.32 | 83.45 | 84.56 | 72.96 |
| B/16·8 `iso_c` | +0.0152 | 90.71 | 90.76 | 90.85 | 90.99 | 90.23 |

Table S29, accuracy as mean ± SD over seeds 0–2 (`--seed 0`, `1`, `2`):

| `g.estimator` | B/32·8 `ta` | B/32·8 `iso_c` | B/32·20 `ta` | B/16·8 `iso_c` |
| :-- | --: | --: | --: | --: |
| `normalized_kl` | 86.96 ± 0.01 | 88.29 ± 0.13 | 84.53 ± 0.11 | 91.10 ± 0.10 |
| `exact_fisher` | 87.26 ± 0.05 | 88.41 ± 0.10 | 84.95 ± 0.03 | 91.18 ± 0.10 |
| `mc_fisher` | 86.87 ± 0.04 | 88.29 ± 0.14 | 84.37 ± 0.06 | 91.05 ± 0.06 |
| `empirical_fisher` | 86.55 ± 0.02 | 88.03 ± 0.11 | 83.55 ± 0.17 | 90.93 ± 0.06 |

Table S32, pipeline columns: the total time is `merge_seconds` and the statistics time the
sum of `statistics_seconds` over `rounds` in `result.json` of the B/32·8 `iso_c` runs above
(and of `--set iteration.rounds=8` for the last row). On a dedicated RTX 4090 the default
recipe took 10.3 min with 1.0 min of statistics, the exact Fisher 29.1 min with 19.7 min,
the Monte Carlo Fisher 11.0 min with 1.6 min, and eight rounds 20.0 min with 2.0 min.

## Table S30 — Shrinkage of G

```bash
for shrinkage in automatic 0.0 0.25 0.5 0.75 1.0; do
  if-merge clip --arch ViT-B-32 --tasks 8 --init ta --set g.shrinkage=$shrinkage
done
```

Expected accuracy, seed 0 (`automatic` is the Ledoit–Wolf estimate of the default recipe;
1.0 is G = I):

| `g.shrinkage` | B/32·8 `ta` | B/32·8 `iso_c` | B/32·20 `ta` | B/16·8 `iso_c` |
| :-- | --: | --: | --: | --: |
| `automatic` | 86.95 | 88.26 | 84.56 | 90.99 |
| 0.0 | 87.06 | 88.54 | 84.77 | 91.29 |
| 0.25 | 87.27 | 88.40 | 84.78 | 91.23 |
| 0.5 | 87.03 | 88.27 | 84.22 | 91.13 |
| 0.75 | 86.75 | 88.05 | 83.34 | 90.87 |
| 1.0 | 85.58 | 87.55 | 78.40 | 90.71 |

## Table S31 — Solver budget and round cap

(a) Cap of the conjugate-gradient iterations per round and the initial guess of the solver
(the current weights, the mean of the expert weights, or zero):

```bash
for iterations in 25 50 100 200 400; do
  if-merge clip --arch ViT-B-32 --tasks 8 --init iso_c --set solver.max_iterations=$iterations
done
if-merge clip --arch ViT-B-32 --tasks 8 --init iso_c --set solver.warm_start=expert_mean
if-merge clip --arch ViT-B-32 --tasks 8 --init iso_c --set solver.warm_start=expert_mean \
  --set solver.max_iterations=400
if-merge clip --arch ViT-B-32 --tasks 8 --init iso_c --set solver.warm_start=zero
```

Expected accuracy, seed 0:

| CG cap / initial guess | B/32·8 `iso_c` | B/32·8 `wa` | B/32·20 `ta` | B/16·8 `iso_c` |
| :-- | --: | --: | --: | --: |
| 25 / current weights | 88.48 | 85.79 | 81.98 | 91.23 |
| 50 / current weights | 88.44 | 86.58 | 83.81 | 91.09 |
| 100 / current weights | 88.26 | 87.02 | 84.56 | 90.99 |
| 200 / current weights | 88.25 | 87.16 | 84.79 | 90.92 |
| 400 / current weights | 88.23 | 87.23 | 84.81 | 90.92 |
| 100 / expert mean | 87.92 | 86.86 | 84.23 | 90.78 |
| 100 / zero | 86.20 | 77.99 | 71.63 | 89.23 |
| 400 / expert mean | 88.16 | 87.22 | 84.75 | 90.94 |

(b) At most eight rounds instead of four (`--set iteration.rounds=8`); the selected round
is in parentheses:

| Setting, initialization | Four rounds | Eight rounds |
| :-- | --: | --: |
| B/32·8 `iso_c` | 88.26 | 88.27 (8) |
| B/32·8 `tsv_m` | 87.80 | 87.82 (8) |
| B/32·8 `ta` | 86.95 | 87.09 (8) |
| B/32·20 `iso_c` | 85.15 | 85.15 (4) |
| B/32·20 `ta` | 84.56 | 84.76 (8) |
| B/16·8 `iso_c` | 90.99 | 90.99 (4) |

---

## Endpoint analyses

The analyses of the IF-Merge endpoints are scripts under [`analysis/`](../analysis); their
commands, inputs and expected numbers are in [`analysis/README.md`](../analysis/README.md).

| Paper item | Script |
| :-- | :-- |
| Table S34, prediction disagreement between initializations and between endpoints | `analysis/prediction_disagreement.py` |
| Table S35, weight distance and input-weighted distance | `analysis/parameter_distances.py` |
| Figure 3, Tables S36 and S37, spectral band projections and band magnitude spread | `analysis/spectral_bands.py` |
| Table 8a, Table S38a,b, IF-Merge from initializations mixed by spectral band | `analysis/mix_initializations.py`, then `if-merge clip --init <checkpoint>` |
| Table S40, linear interpolation paths | `analysis/interpolation_paths.py` |
| Table S41, accuracy on unseen tasks | `analysis/unseen_tasks.py` |

The jointly trained models of Tables S35, S36, S40 and S41 come from `if-merge joint` with
`--seed 0`, `1`, `2`, with `--init wa` and `--init iso_c`, and with
`--save-steps 200 500 1000 1500 2000` for the intermediate checkpoints of Table S41.

## Not covered by this release

The following CLIP items are analyses of saved models or instrumented runs. The models
they analyze are produced by the commands above; the analysis scripts are not included.

| Paper item | What it needs |
| :-- | :-- |
| Figure 1 | `figures/teaser/make_teaser.py` draws the landscape from the predictions stored next to it; the script that collects these predictions is not included. |
| Figure 2, Figures S1–S9, Tables S1–S3 (measurements for the analysis section) | Layerwise probes of the statistics and of candidate updates at fixed models, and two runs instrumented in every round. Figure S8 only plots `block_steps` of the Table 2 runs from `ta` and `iso_cts`. |
| Figure 3 (the plot) | `analysis/spectral_bands.py` gives its numbers; the plotting script is not included. |
| Table 8b, Table S38c (transplants between endpoints) | A script that adds or removes the head, the tail or the full difference of two endpoints and evaluates the result. |
| Table S16 (cost) | Wall time and peak memory of every method measured on a dedicated GPU; each command records its own run time and peak GPU memory in `result.json`. |
| Table S26 (local objective and full-model response) | Evaluation of single-block and full-model candidates at several step sizes in every round. |
| Table S28, columns "G heterogeneity" and "Input overlap" | Statistics of the first-round A and G. |
| Table S32, single-collection columns | Repeated timing of one statistics collection per construction. |
| Table S39 (band retention under solver settings) | The merged model after every round (`--save-rounds`) and band profiles restricted to the blocks that accepted full steps. |
| Tables S17–S19 (additional autoregressive models) | The pipeline for these models is not part of this release. |
