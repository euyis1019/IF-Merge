# Endpoint analyses

Scripts behind the analyses of the IF-Merge endpoints (Sec. 5.4 and the supplementary
material of the paper). They read merged models that `if-merge` has written and import the
`if_merge` package; run them from the repository root after `pip install -e .`.

| Script | Paper item | What it computes | Cost (ViT-B/32, 8 tasks) |
| :-- | :-- | :-- | :-- |
| [`prediction_disagreement.py`](prediction_disagreement.py) | Table S34 | top-1 disagreement between models on 1,024 test images per task | GPU, 20 s per model |
| [`parameter_distances.py`](parameter_distances.py) | Table S35 | weight distance and input-weighted distance, blocks 2–11 | GPU, 3 min for 33 models |
| [`spectral_bands.py`](spectral_bands.py) | Fig. 3, Tables S36, S37 | band projection coefficients, band magnitude spread | CPU, 10 min for 18 pairs |
| [`mix_initializations.py`](mix_initializations.py) | Table 8a, Table S38a,b | initializations mixed by spectral band, for `if-merge clip --init` | CPU, 2 min |
| [`interpolation_paths.py`](interpolation_paths.py) | Table S40 | accuracy and loss barriers on linear paths | GPU, 2 min per path |
| [`unseen_tasks.py`](unseen_tasks.py) | Table S41 | accuracy on the merged and on the unseen tasks | GPU, 15 s per model |

`python analysis/<script>.py --help` prints the definitions; the module docstrings state
exactly what is computed where the paper describes it more briefly.

## Checkpoints

Every model is an open_clip visual state dict and is passed as `NAME=CHECKPOINT`, where
`CHECKPOINT` is

- the path of a `merged.pt` written by `if-merge clip`, `if-merge distill` or
  `if-merge baseline`, or of a `model.pt` / `step<N>.pt` written by `if-merge joint`;
- a baseline id (`wa`, `ta`, `ties`, `regmean`, `adamerging`, `regmeanpp`, `tsv_m`, `wudi`,
  `iso_c`, `iso_cts`), built like the initialization of `if-merge clip --init`;
- `pretrained`.

The examples use ViT-B/32 with 8 tasks and these runs (see
[`docs/REPRODUCE.md`](../docs/REPRODUCE.md) for their expected accuracies):

```bash
B="--arch ViT-B-32 --tasks 8"
E=runs/clip/ViT-B-32-8tasks/if_merge     # IF-Merge endpoints:      $E/<init>-seed<K>/merged.pt
K=runs/clip/ViT-B-32-8tasks/distill      # distillation:            $K/<init>-seed0-n512/merged.pt
J=runs/clip/ViT-B-32-8tasks/joint        # jointly trained models:  $J/[<init>-]steps3000-seed<K>/model.pt

for init in wa ta ties regmean adamerging regmeanpp tsv_m wudi iso_c iso_cts; do
  if-merge clip $B --init $init
done
for init in wa ta regmean tsv_m wudi iso_c; do
  if-merge distill $B --init $init
done
if-merge prepare-clip --arch ViT-B-32 --tasks 8 --train
for seed in 0 1 2; do
  if-merge joint $B --seed $seed --save-steps 200 500 1000 1500 2000    # also writes step<N>.pt
done
if-merge joint $B --init wa
if-merge joint $B --init iso_c
```

All scripts take `--data`, `--runs` and `--device` like `if-merge`, and `--out FILE` to
write the result as JSON.

## Prediction disagreement — Table S34

```bash
INITS="wa ta ties regmean adamerging regmeanpp tsv_m wudi iso_c iso_cts"
G="--group ordinary=wa,ta,ties,regmean,adamerging,regmeanpp --group spectral=iso_c,iso_cts --group subspace=tsv_m,wudi"
python analysis/prediction_disagreement.py $B $G --models $INITS                                    # initializations
python analysis/prediction_disagreement.py $B $G --models $(for i in $INITS; do echo $i=$E/$i-seed0/merged.pt; done)
```

The first command gives the "Ordinary, initial" column (group `ordinary|ordinary`), the
second the other columns. Expected (%):

| Setting | Initializations | Ordinary, initial | Ordinary, final | Ordinary–spectral | Iso-C–Iso-CTS | Ordinary–subspace | Subspace–spectral |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| B/32·8 | 10 | 19.4 | 3.0 | 5.7 | 2.8 | 4.0 | 4.5 |
| B/32·14 | 3 | 7.5 | 2.0 | 7.7 | – | – | – |
| B/32·20 | 9 | 22.5 | 3.2 | 5.7 | 3.7 | 3.5 | 5.0 |
| B/16·8 | 10 | 15.4 | 2.1 | 4.6 | 2.0 | 3.5 | 3.3 |
| B/16·14 | 3 | 14.5 | 1.7 | 7.1 | – | – | – |
| B/16·20 | 3 | 14.0 | 2.3 | 6.4 | – | – | – |
| L/14·8 | 3 | 13.0 | 0.9 | 4.2 | – | – | – |
| L/14·14 | 3 | 14.5 | 1.1 | 5.0 | – | – | – |
| L/14·20 | 3 | 12.2 | 1.1 | 5.0 | – | – | – |

Settings with three initializations use `wa`, `ta` and `iso_cts`. On B/32·20, `ta` is the
same model as `wa` and is left out. The means of the two "Ordinary" columns over the nine
settings are the 14.8% and 1.9% quoted in the main text.

## Parameter distances — Table S35

```bash
python analysis/parameter_distances.py $B --models \
  $(for i in $INITS; do echo init_$i=$i if_$i=$E/$i-seed0/merged.pt; done) \
  $(for i in wa ta regmean tsv_m wudi iso_c; do echo kd_$i=$K/$i-seed0-n512/merged.pt; done) \
  if_wa_seed1=$E/wa-seed1/merged.pt if_wa_seed2=$E/wa-seed2/merged.pt \
  if_iso_c_seed1=$E/iso_c-seed1/merged.pt if_iso_c_seed2=$E/iso_c-seed2/merged.pt \
  joint0=$J/steps3000-seed0/model.pt joint1=$J/steps3000-seed1/model.pt joint2=$J/steps3000-seed2/model.pt
```

The script prints both distances for every pair. A row of the table is the median (range)
over a set of pairs; expected for B/32·8:

| Row | Pairs | Weights | Input-weighted |
| :-- | :-- | --: | --: |
| Between experts | printed as `between experts` (28) | 1.41 (1.24–1.59) | 1.40 (1.24–1.50) |
| Ordinary four, initial states | `init_a` \| `init_b` for a, b in wa, ta, ties, adamerging (6) | 0.80 (0.38–0.94) | 0.75 (0.39–1.17) |
| Ordinary four, endpoints | `if_a` \| `if_b` for the same (6) | 1.14 (1.04–1.34) | 0.24 (0.22–0.29) |
| With spectral or subspace init., endpoints | `if_a` \| `if_b` with a or b in tsv_m, wudi, iso_c, iso_cts and neither in the RegMean family (22) | 2.28 (1.74–3.20) | 0.90 (0.69–1.35) |
| With RegMean family, initial states | `init_a` \| `init_b` with a or b in regmean, regmeanpp (17) | 7.58 (4.89–8.36) | 0.98 (0.29–1.75) |
| With RegMean family, endpoints | `if_a` \| `if_b` for the same (17) | 7.07 (4.59–7.74) | 0.36 (0.26–1.30) |
| Same init., different calibration images | `if_wa` \| `if_wa_seed1`, `if_wa_seed2`; `if_iso_c` \| `if_iso_c_seed1`, `if_iso_c_seed2` (4) | 1.66 (1.30–1.86) | 0.39 (0.37–0.42) |
| Initialization to its IF endpoint | `init_a` \| `if_a` (10) | 2.20 (1.42–2.62) | 0.70 (0.31–1.26) |
| Initialization to its KD endpoint | `init_a` \| `kd_a` (6) | 0.17 (0.12–0.37) | 0.20 (0.14–0.43) |
| IF endpoints vs. joint training seed 0 | `if_a` \| `joint0`, a outside the RegMean family (8) | 2.73 (2.57–3.20) | 1.47 (1.45–1.75) |
| Joint training vs. pretrained | `pretrained` \| `joint0`, `joint1`, `joint2` (3) | 1.28 (1.28) | 1.31 (1.31) |

The seed-level part of the table (seeds 1 and 2) uses the endpoints of `--seed 1` and
`--seed 2` in place of the seed-0 endpoints; the initializations, the reference images and
the normalization do not change.

## Spectral bands — Fig. 3, Tables S36, S37

```bash
python analysis/spectral_bands.py $B \
  --models $(for i in $INITS; do echo $i=$i:$E/$i-seed0/merged.pt; done) \
           $(for i in wa ta regmean tsv_m wudi iso_c; do echo kd_$i=$i:$K/$i-seed0-n512/merged.pt; done) \
           joint_iso_c=iso_c:$J/iso_c-steps3000-seed0/model.pt joint_wa=wa:$J/wa-steps3000-seed0/model.pt \
  --pairs iso_c:wa iso_c:ta iso_c:ties iso_c:regmean wudi:wa tsv_m:wa wa:ta wa:regmean ta:regmean \
          kd_iso_c:kd_wa kd_iso_c:kd_ta kd_iso_c:kd_regmean kd_wudi:kd_wa kd_tsv_m:kd_wa kd_wa:kd_ta \
          kd_wa:kd_regmean kd_ta:kd_regmean joint_iso_c:joint_wa \
  --spread wa ta ties regmean adamerging regmeanpp
```

Table S36, band projection coefficients (median over 74 matrices):

| Pair | Top 1% | 1–10% | 10–50% | Bottom 50% | KD, top 1% | KD, bottom 50% |
| :-- | --: | --: | --: | --: | --: | --: |
| Iso-C–WA | −0.08 | 0.16 | 0.44 | 0.67 | 0.92 | 1.00 |
| Iso-C–TA | 0.09 | 0.26 | 0.50 | 0.68 | 0.97 | 1.00 |
| Iso-C–TIES | 0.18 | 0.39 | 0.51 | 0.68 | – | – |
| Iso-C–RegMean | 0.56 | 0.52 | 0.58 | 0.77 | 0.99 | 1.00 |
| WUDI–WA | 0.05 | 0.13 | 0.24 | 0.52 | 0.93 | 1.00 |
| TSV-M–WA | 0.01 | 0.04 | 0.15 | 0.45 | 0.92 | 1.00 |
| WA–TA | −0.04 | −0.02 | 0.00 | 0.05 | 0.86 | 0.98 |
| WA–RegMean | 0.01 | 0.05 | 0.17 | 0.58 | 0.91 | 1.00 |
| TA–RegMean | 0.11 | 0.12 | 0.23 | 0.59 | 0.98 | 1.00 |
| Joint training (`joint_iso_c:joint_wa`) | 0.58 | 0.82 | 0.94 | 0.99 | | |
| Random bands (`random_bands` of `iso_c:wa`) | 0.49 | 0.49 | 0.50 | 0.50 | | |

Fig. 3 plots the four band coefficients of the pairs with `wa`. Table S37 is the `spread`
entry (`--spread` lists the ordinary initializations of the setting; `--permutations 0`
skips the random bands):

| Setting | Models in `--spread` | Initializations: top 1% | bottom 50% | Endpoints: top 1% | bottom 50% |
| :-- | :-- | --: | --: | --: | --: |
| B/32·8 | wa ta ties regmean adamerging regmeanpp | 2.57 | 2.60 | 1.09 | 1.08 |
| B/32·14 | wa ta | 1.40 | 1.40 | 1.03 | 1.02 |
| B/32·20 | wa ta ties regmean adamerging regmeanpp | 3.74 | 3.74 | 1.14 | 1.22 |
| B/16·8 | wa ta ties regmean adamerging regmeanpp | 2.88 | 2.68 | 1.09 | 1.16 |
| B/16·14 | wa ta | 2.10 | 2.10 | 1.05 | 1.11 |
| B/16·20 | wa ta | 2.00 | 2.00 | 1.02 | 1.10 |
| L/14·8 | wa ta | 2.40 | 2.40 | 1.02 | 1.11 |
| L/14·14 | wa ta | 2.80 | 2.80 | 1.04 | 1.13 |
| L/14·20 | wa ta | 2.00 | 2.00 | 1.03 | 1.07 |

## Mixed initializations — Table 8a, Table S38a,b

```bash
python analysis/mix_initializations.py $B --a wa --b iso_c --blocks 4 6 8 9 10 11 --out-dir runs/mixed
for seed in 0 1 2; do
  for mix in wa_head+iso_c_tail iso_c_head+wa_tail wa_head+random; do
    if-merge clip $B --init runs/mixed/$mix.pt --seed $seed
  done
done
```

Expected accuracy / holdout KL of the selected round after IF-Merge:

| Initialization | File | Seed 0 | Seed 1 | Seed 2 | Accuracy, mean ± SD | Holdout KL, mean ± SD |
| :-- | :-- | --: | --: | --: | --: | --: |
| WA | `wa` | 87.02 / 0.2063 | 87.02 / 0.1806 | 86.95 / 0.1987 | 87.00 ± 0.04 | 0.1952 ± 0.0132 |
| Iso-C | `iso_c` | 88.26 / 0.1124 | 88.43 / 0.0890 | 88.17 / 0.1110 | 88.29 ± 0.13 | 0.1041 ± 0.0131 |
| WA-H + Iso-C-T | `wa_head+iso_c_tail.pt` | 87.60 / 0.1458 | 87.81 / 0.1203 | 87.67 / 0.1373 | 87.69 ± 0.11 | 0.1345 ± 0.0130 |
| Iso-C-H + WA-T | `iso_c_head+wa_tail.pt` | 87.01 / 0.2048 | 87.12 / 0.1767 | 86.90 / 0.1947 | 87.01 ± 0.11 | 0.1921 ± 0.0142 |
| WA-H + Random | `wa_head+random.pt` | 86.94 / 0.2110 | 86.98 / 0.1804 | 86.86 / 0.2032 | 86.93 ± 0.06 | 0.1982 ± 0.0159 |

Table S38a, seed 0:

| Initialization | File | Holdout KL | Accuracy |
| :-- | :-- | --: | --: |
| WA linear layers + Iso-C other parameters | `wa_linear+iso_c_other.pt` | 0.2088 | 87.00 |
| Iso-C linear layers + WA other parameters | `iso_c_linear+wa_other.pt` | 0.1105 | 88.35 |
| WA with six Iso-C blocks | `wa+iso_c_blocks_4-6-8-9-10-11.pt` | 0.1200 | 88.32 |
| WA with eight Iso-C blocks | `--blocks 2 4 6 7 8 9 10 11` | 0.1156 | 88.22 |

The mixed initializations are computed in float64 and stored in float32; a rebuilt file can
differ from another machine's in the last bit of a few weights, which moves the numbers
above in the fourth digit (seed 0 of the first row: 87.60 / 0.1457). The transplants between
endpoints (Table 8b, Table S38c) are not covered.

## Interpolation paths — Table S40

```bash
python analysis/interpolation_paths.py $B \
  --models if_wa=$E/wa-seed0/merged.pt if_ta=$E/ta-seed0/merged.pt if_ties=$E/ties-seed0/merged.pt \
           if_regmean=$E/regmean-seed0/merged.pt if_iso_c=$E/iso_c-seed0/merged.pt ta=ta \
           joint0=$J/steps3000-seed0/model.pt joint1=$J/steps3000-seed1/model.pt joint2=$J/steps3000-seed2/model.pt \
           joint_iso_c=$J/iso_c-steps3000-seed0/model.pt \
  --paths if_ta:if_wa if_ta:if_ties if_wa:if_ties if_ta:if_regmean if_iso_c:if_wa \
          if_ta:joint0 if_wa:joint1 if_ties:joint2 if_regmean:joint0 if_ta:joint_iso_c if_iso_c:joint0 \
          joint0:joint1 joint0:joint2 ta:joint0
```

Expected (barriers in percentage points; the second worst-task column is the same command
with `--images 256`; expert KL at the first model / minimum on the path / at the second):

| Path | CE barrier | Accuracy barrier | Worst task, 1,024 | Worst task, 256 | Expert KL |
| :-- | --: | --: | --: | --: | :-- |
| `if_ta:if_wa` | 0.0000 | 0.00 | 0.31 | 0.66 | 0.223 / 0.222 / 0.223 |
| `if_ta:if_ties` | 0.0000 | 0.07 | 0.49 | 1.41 | 0.223 / 0.218 / 0.218 |
| `if_wa:if_ties` | 0.0000 | 0.04 | 0.47 | 0.66 | 0.223 / 0.218 / 0.218 |
| `if_ta:if_regmean` | 0.0000 | 0.05 | 0.51 | 1.33 | 0.223 / 0.214 / 0.217 |
| `if_iso_c:if_wa` | 0.0000 | 0.00 | 0.54 | 1.48 | 0.189 / 0.188 / 0.223 |
| `if_ta:joint0` | 0.0000 | 0.00 | 0.20 | 0.51 | 0.223 / 0.173 / 0.211 |
| `if_wa:joint1` | 0.0000 | 0.00 | 0.10 | 0.47 | 0.223 / 0.176 / 0.225 |
| `if_ties:joint2` | 0.0000 | 0.00 | 0.08 | 0.62 | 0.218 / 0.176 / 0.225 |
| `if_regmean:joint0` | 0.0000 | 0.00 | 0.13 | 1.52 | 0.217 / 0.170 / 0.211 |
| `if_ta:joint_iso_c` | 0.0000 | 0.00 | 0.00 | 0.31 | 0.223 / 0.147 / 0.198 |
| `if_iso_c:joint0` | 0.0000 | 0.00 | 0.03 | 0.62 | 0.189 / 0.159 / 0.211 |
| `joint0:joint1` | 0.0000 | 0.00 | 0.24 | 0.66 | 0.211 / 0.203 / 0.225 |
| `joint0:joint2` | 0.0000 | 0.00 | 0.29 | 1.21 | 0.211 / 0.201 / 0.225 |
| `ta:joint0` | 0.0000 | 0.00 | 0.00 | 0.00 | 0.929 / 0.203 / 0.211 |

## Unseen tasks — Table S41

Needs the heads and test images of all 20 tasks (`if-merge prepare-clip --tasks 20`).

```bash
python analysis/unseen_tasks.py $B \
  --models pretrained ta ties regmean iso_c \
           $(for i in wa ta ties regmean iso_c iso_cts; do echo if_$i=$E/$i-seed0/merged.pt; done) \
           $(for n in 200 500 1000 1500 2000; do echo joint0_step$n=$J/steps3000-seed0/step$n.pt; done) \
           joint0=$J/steps3000-seed0/model.pt joint1=$J/steps3000-seed1/model.pt joint2=$J/steps3000-seed2/model.pt \
           joint_wa=$J/wa-steps3000-seed0/model.pt joint_iso_c=$J/iso_c-steps3000-seed0/model.pt \
  --mix mid_ta=if_ta:joint0:0.5 mid_iso_c=if_iso_c:joint0:0.5 \
        $(for w in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do echo wise_$w=pretrained:joint0:$w; done)
```

Expected accuracy (%) on the 8 merged tasks and on the 12 unseen tasks, 256 test images per
task (the paper prints one decimal and omits some merged-task cells):

| Model | Merged | Unseen | | Model | Merged | Unseen |
| :-- | --: | --: | --- | :-- | --: | --: |
| `pretrained` | 49.32 | 62.57 | | `joint0_step200` | 77.44 | 58.14 |
| `ta` | 69.92 | 54.59 | | `joint0_step500` | 82.67 | 54.46 |
| `ties` | 72.90 | 56.25 | | `joint0_step1000` | 85.21 | 54.95 |
| `regmean` | 84.08 | 57.71 | | `joint0_step1500` | 87.70 | 52.02 |
| `iso_c` | 84.28 | 53.81 | | `joint0_step2000` | 88.57 | 53.26 |
| `if_wa` | 87.45 | 57.13 | | `joint0` | 90.14 | 52.99 |
| `if_ta` | 87.01 | 56.48 | | `joint1` | 89.75 | 51.69 |
| `if_ties` | 87.40 | 56.80 | | `joint2` | 90.33 | 53.19 |
| `if_regmean` | 87.65 | 56.48 | | `joint_wa` | 90.23 | 53.12 |
| `if_iso_c` | 88.57 | 57.62 | | `joint_iso_c` | 89.89 | 51.37 |
| `if_iso_cts` | 89.36 | 56.22 | | `wise_0.3` | 77.20 | 62.30 |
| `mid_ta` | 89.79 | 58.01 | | `wise_0.5` | 86.28 | 60.71 |
| `mid_iso_c` | 90.53 | 57.62 | | `wise_0.7` | 89.11 | 58.79 |
| | | | | `wise_0.9` | 90.19 | 55.44 |

The intermediate checkpoints `step<N>.pt` are snapshots of the 3,000-step run, whose
learning-rate schedule spans all 3,000 steps; they are not runs with a shorter budget.
For `if_regmean` the paper prints 56.4 on the unseen tasks: one test image is classified
differently there.
