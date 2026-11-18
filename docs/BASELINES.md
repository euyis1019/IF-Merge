# Baselines

Every baseline of the paper's CLIP tables is reimplemented in `src/if_merge/baselines/`
(the algorithms) and run by `src/if_merge/clip/baselines.py` (data, selection, evaluation).

```bash
if-merge baseline --list
if-merge baseline --arch ViT-B-32 --tasks 8 --method tsv_m
if-merge baseline --arch ViT-B-32 --tasks 8 --method reforge --init iso_cts   # post-hoc
```

A run writes `result.json` (validation sweep, selected hyperparameters, test accuracy per
task, seconds, peak GPU memory) and `merged.pt` (open_clip visual state dict) to
`runs/clip/<arch>-<T>tasks/baselines/<method>[+<init>]/`. The numbers of the paper are in
`results/clip_baselines.json`, the selected hyperparameters in
`configs/baselines/clip_selected.yaml`, the grids in `configs/baselines/clip.yaml`.

## Protocol

- **Selection.** Hyperparameters with a grid are selected on the labeled validation split
  by task-averaged accuracy (ties go to the earlier grid point). The selected model is
  evaluated once on the test split. Methods without a grid are evaluated once.
- **Evaluation.** Top-1 accuracy (%) with the frozen classification heads under float16
  autocast, averaged over tasks, as for IF-Merge.
- **Unlabeled calibration images.** Methods that need data read the first images of the
  same calibration draw as IF-Merge (`images/<Task>/calib_seed<K>.pt`, 512 training images
  per task). Labels are never used.
- **Scaling coefficient.** For task-vector methods the merged model is
  `theta_0 + scale * tau`, where `tau` is the merged task vector; `scale` is on the grid.
- **Two layouts of the encoder.** In the open_clip state dict the query, key and value
  projections of a block are one packed `3d x d` matrix (`attn.in_proj_weight`).
  Data-free methods and LiNeS/SVC work on this *packed* layout. Methods that read layer
  inputs work on the *split* layout, where q, k and v are separate `d x d` matrices.
  For SVD-based rules the two are different methods: one SVD of the packed matrix is not
  three SVDs of its parts. A merged model is evaluated in the layout it was built in. The
  two layouts compute the same function, but under float16 autocast they can disagree on
  a few test images (Weight Averaging on ViT-B/32, 8 tasks: 65.49 packed, 65.49 split,
  65.50 with all linear maps exposed as used by IF-Merge).
- **Matrices.** SVD-based rules (TSV-M, Iso-C, Iso-CTS, WUDI, SVC) treat every
  two-dimensional tensor as a task matrix. In the packed layout these are, per block,
  `in_proj_weight`, `out_proj.weight` and the two MLP weights, plus the positional
  embedding and the output projection `proj`. The patch embedding (4-D), class embedding,
  LayerNorms and biases are not matrices.

## Methods

| id | Method | Data | Selected on validation |
| --- | --- | --- | --- |
| `wa` | Weight Averaging | none | nothing |
| `ta` | Task Arithmetic | none | scale in 0.05..0.60 (step 0.05) |
| `ties` | TIES | none | scale in 0.05..0.60 |
| `tsv_m` | TSV-M | none | scale in 0.3..2.0 (step 0.1) |
| `wudi` | WUDI | none | scale in 0.3..2.0 |
| `iso_c` | Iso-C | none | scale in 0.3..2.0 |
| `iso_cts` | Iso-CTS | none | scale in 0.3..2.0 |
| `regmean` | RegMean | 256 unlabeled images per task | off-diagonal scale in {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} |
| `regmeanpp` | RegMean++ | 256 unlabeled images per task | off-diagonal scale in {0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0} |
| `fisher` | Fisher Merging | 256 unlabeled images per task | nothing |
| `mats` | MaTS | 256 unlabeled images per task | the scale of its Task Arithmetic initialization |
| `adamerging` | AdaMerging (layer-wise) | 512 unlabeled images per task | nothing |
| `lines` | LiNeS (post-hoc) | none | scale in 0.05..1.50 (step 0.05) |
| `svc` | SVC (post-hoc) | none | strength in {0.1, 0.2, 0.5, 1.0} x scale in 0.3..2.0 |
| `reforge` | ReForge (post-hoc) | 128 unlabeled images per task | 10 hyperparameters, 200 validation evaluations |

### Weight Averaging (`wa`)

Wortsman et al., *Model Soups*, ICML 2022 ([code](https://github.com/mlfoundations/model-soups)).
The plain mean of the expert weights over all tensors of the visual encoder.

### Task Arithmetic (`ta`)

Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023
([code](https://github.com/mlfoundations/task_vectors)). Sum of the task vectors, one
scaling coefficient for all tensors.

### TIES (`ties`)

Yadav et al., *TIES-Merging*, NeurIPS 2023 ([code](https://github.com/prateeky2806/ties-merging)).
Each task vector keeps its top 20% entries by magnitude, ranked over the whole flattened
visual encoder; signs are elected by the summed trimmed values. Deviation: the agreeing
entries are summed (`dis-sum`, as in the AdaMerging, TSV-M and Iso-C code bases) instead
of averaged, and the scaling coefficient is selected.

### TSV-M (`tsv_m`)

Gargiulo et al., *Task Singular Vectors*, CVPR 2025
([code](https://github.com/AntoAndGar/task_singular_vectors)). Follows the official
implementation on the packed layout: per matrix, the top `1/T` singular directions of
every task, concatenated and orthogonalized; other tensors take the mean of the task
vectors. Deviation: the scaling coefficient is selected instead of fixed.

### Iso-C and Iso-CTS (`iso_c`, `iso_cts`)

Marczak et al., *No Task Left Behind: Isotropic Model Merging with Common and
Task-Specific Subspaces*, ICML 2025 ([code](https://github.com/danielm1405/iso-merging)).
Follows the official implementation on the packed layout; Iso-CTS uses the official
common-subspace fraction 0.8. Other tensors take the mean of the task vectors.

### WUDI (`wudi`)

Cheng et al., *Whoever Started the Interference Should End It*, ICML 2025
([code](https://github.com/nathanielyvo/WUDI-Merging)). 300 Adam steps with learning rate
1e-5 per matrix, started from the Task Arithmetic matrix (official ViT settings).
Deviations: all two-dimensional tensors are optimized and the remaining tensors take the
mean of the task vectors (the official ViT script optimizes the attention and MLP
weights and leaves everything else at the pretrained value), and the scaling coefficient
is selected instead of fixed to 1.

### RegMean (`regmean`)

Jin et al., *Dataless Knowledge Fusion by Merging Weights of Language Models*, ICLR 2023
([code](https://github.com/bloomberg/dataless-model-merging)). Split layout. The six
linear maps of every block (q, k, v, attention output, two MLP layers) are solved in
closed form from the Gram matrices of their inputs, measured in each expert on 256
images of its own task; all other tensors are the expert mean. The scale applied to the
off-diagonal Gram entries is selected (the original uses 0.1; 1.0 means no shrinkage).
A grid point whose linear system is singular is rejected.

### RegMean++ (`regmeanpp`)

Nguyen et al., *RegMean++*, TMLR 2026 ([code](https://github.com/nthehai01/RegMean-plusplus)).
Follows the official CLIP implementation: blocks are merged in forward order; the inputs
produced by the merged prefix are passed through each expert's block to collect the Gram
matrices; linear weights are solved as in RegMean (float64), block biases and LayerNorms
are averaged. Tensors outside the blocks are the expert mean; this includes the output
projection `proj`, which the official Hugging Face model does not contain. The
off-diagonal scale is selected (the official CLIP default is 0.95); every grid point is a
full forward merge.

### Fisher Merging (`fisher`)

Matena and Raffel, *Merging Models with Fisher-Weighted Averaging*, NeurIPS 2022
([code](https://github.com/mmatena/model_merging)). We use the estimator of the
[FusionBench](https://github.com/tanganke/fusion_bench) CLIP implementation: on 256
images per task in batches of 32, `sum_i sum_c sqrt(p_ic) log p_ic` is back-propagated
once per batch and the squared batch gradient is accumulated (a diagonal Fisher estimate
that needs no labels and depends on the batch size). A floor of 1e-6 is added to every
entry, and experts are weighted by the inverse L2 norm of their Fisher. All parameters of
the split layout are merged this way. Nothing is selected.

### MaTS (`mats`)

Tam et al., *Merging by Matching Models in Task Parameter Subspaces*, TMLR 2024
([code](https://github.com/r-three/mats)). The RegMean objective (same Gram matrices,
no off-diagonal scaling) solved with 100 conjugate-gradient steps per weight matrix.
The solver stops before convergence, so the initialization matters: we start from Task
Arithmetic with its validation-selected scaling coefficient (the official recipe uses the
fixed coefficient 0.3). Tensors that are not linear weights keep their Task Arithmetic
value. `run_baseline` runs the Task Arithmetic sweep itself; the sweep and
`validation_average` in its `result.json` are those of the Task Arithmetic initialization.

### AdaMerging (`adamerging`)

Yang et al., *AdaMerging*, ICLR 2024 ([code](https://github.com/EnnengYang/AdaMerging)).
Layer-wise variant: one coefficient per task and per tensor, initialized to 0.3, clamped
to [0, 1], trained with Adam (learning rate 1e-3, 500 steps, 16 images per task and step)
to minimize the prediction entropy. Deviations: the entropy is computed on the 512
unlabeled calibration images per task instead of unlabeled test images, the forward pass
uses float16 autocast, and the model after the last step is reported. The image sampling
is seeded (`--seed`). The merged model is evaluated on the frozen encoder used during
training; float16 autocast treats frozen and trainable weights slightly differently, which
moves the test accuracy by about 0.01 (80.05 instead of 80.06 on ViT-B/32 with 8 tasks).

### LiNeS (`lines`, post-hoc)

Wang et al., *LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model
Merging*, ICLR 2025 ([code](https://github.com/wang-kee/LiNeS)). The merged task vector of
the initialization is scaled by `scale * (1 + l / (L - 1))` in block `l`; tensors outside
the blocks are scaled by `scale`. Deviation: we fix the ratio between the slope and the
intercept of the paper's schedule to 1 and select only `scale`.

### SVC (`svc`, post-hoc)

*When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging*,
arXiv:2602.05536 ([code](https://github.com/lyymuwu/SVC)). Follows the accelerated
FusionBench implementation (float64) on all two-dimensional tensors of the packed layout.
The scaling coefficient of the initialization is applied before the calibration.
Deviation: the calibration strength is selected jointly with the scaling coefficient
instead of being fixed per initialization. For an initialization with a flat spectrum
(Iso-C, Iso-CTS) the singular basis, and with it the result, is not unique; we use the
basis returned by one SVD of the merged matrix.

### ReForge (`reforge`, post-hoc)

Li et al., *ReForge: Refining Merged Models with Anchor-Regularized Regression*,
arXiv:2605.12843. **This is our reimplementation and only approximates the method.**
The inner solve is the paper's closed form: RegMean with a ridge that pulls each weight
matrix toward the initialization (anchor), from Gram matrices measured once in the
experts on 128 images per task. The outer search differs from the paper's 15-dimensional
search space:

- 10 hyperparameters: for each of three depth groups (shallow, middle, deep blocks) the
  log10 ridge relative to the mean diagonal of the summed Gram matrix (-6..2), the
  off-diagonal scale (0..1) and a scale of the solved task vector (0.5..1.5); plus one
  scale of the anchor's task vector (0..1.5).
- 200 evaluations on the labeled validation split, proposed by Gaussian-process Bayesian
  optimization (Matern 5/2 kernel, expected improvement over 4096 random candidates, 20
  random initial points; scikit-learn and scipy are required).
- All six linear maps of every block are solved, in the split layout, from unnormalized
  Gram matrices. Other tensors are the expert mean, not the anchor.
- The anchor is the merged task vector of the initialization computed in the split
  layout, so for TSV-M, WUDI and Iso-CTS it is not identical to the standalone baseline;
  its scaling coefficient is part of the search.

The search is seeded; `--seed K` also selects calibration draw K. With the same seed,
software versions and hardware the search is reproducible; otherwise a single changed
validation score changes all later proposals. The paper reports mean and standard
deviation over seeds.

## Initializations of post-hoc methods

`--init` takes a baseline id or a checkpoint. For a data-free baseline (`wa`, `ta`, `ties`,
`tsv_m`, `wudi`, `iso_c`, `iso_cts`) the post-hoc method starts from that method's merged
task vector and selects the scaling coefficient itself, jointly with its own
hyperparameters. For a checkpoint or any other baseline, the task vector is the difference
between the merged model and the pretrained model.

## Cost

Wall-clock time and peak GPU memory of one complete run (selection and test evaluation
included) on ViT-B/32 with 8 tasks, one RTX 4090; evaluating the test split alone takes
about one minute.

| Method | Time | Peak GPU memory |
| --- | ---: | ---: |
| Task Arithmetic | 1.6 min | 4.7 GB |
| TIES | 1.8 min | 4.7 GB |
| TSV-M | 2.2 min | 4.5 GB |
| WUDI | 3.0 min | 5.6 GB |
| Iso-CTS | 2.4 min | 4.7 GB |
| Fisher Merging | 1.4 min | 5.2 GB |
| RegMean | 4.7 min | 5.0 GB |
| MaTS (given the Task Arithmetic scale) | 1.5 min | 4.5 GB |
| RegMean++ | 7.1 min | 6.5 GB |
| AdaMerging | 5.9 min | 10.1 GB |
| ReForge | 25.5 min | 5.0 GB |
