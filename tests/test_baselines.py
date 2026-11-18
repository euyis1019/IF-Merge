"""Closed forms and invariants of the baseline algorithms on tiny random tensors (CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from if_merge.baselines import BASELINES
from if_merge.baselines.adamerging import adamerging
from if_merge.baselines.fisher import diagonal_fisher, fisher_merge, mats
from if_merge.baselines.posthoc import SingularValueCalibration, lines
from if_merge.baselines.reforge import GaussianProcessOptimizer, anchored_regression, depth_groups
from if_merge.baselines.regmean import regmean, regmean_plus_plus, scale_off_diagonal
from if_merge.baselines.spectral import iso_c, iso_cts, tsv_m, wudi
from if_merge.baselines.weight_space import task_arithmetic, ties, weight_average
from if_merge.clip import baselines as clip_baselines

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "baselines"


def random_task_vectors(count: int = 3, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    shapes = {"layer.weight": (12, 8), "layer.bias": (12,), "embedding": (4, 3, 2)}
    return [
        {name: torch.randn(shape, generator=generator) for name, shape in shapes.items()}
        for _ in range(count)
    ]


def assert_same_shapes(output: dict, reference: dict) -> None:
    assert list(output) == list(reference)
    assert all(output[name].shape == reference[name].shape for name in reference)


# ------------------------------------------------------------- weight space
def test_weight_average_and_task_arithmetic_are_mean_and_sum():
    task_vectors = random_task_vectors()
    mean, total = weight_average(task_vectors), task_arithmetic(task_vectors)
    for name in task_vectors[0]:
        stacked = torch.stack([tv[name] for tv in task_vectors])
        assert torch.equal(mean[name], stacked.mean(0))
        assert torch.equal(total[name], stacked.sum(0))


def test_task_arithmetic_with_one_task_is_the_task_vector():
    (task_vector,) = random_task_vectors(1)
    merged = task_arithmetic([task_vector])
    assert all(torch.equal(merged[name], task_vector[name]) for name in task_vector)


def test_ties_with_identical_vectors_sums_the_trimmed_vector():
    (task_vector,) = random_task_vectors(1)
    merged = ties([task_vector] * 4, retained_percentage=20.0)
    assert_same_shapes(merged, task_vector)
    flat = torch.cat([tensor.reshape(-1) for tensor in task_vector.values()])
    kept = int(flat.numel() * 0.2)
    threshold = flat.abs().sort().values[-kept - 1]
    expected = 4 * flat * (flat.abs() > threshold)
    assert torch.allclose(torch.cat([tensor.reshape(-1) for tensor in merged.values()]), expected)
    assert int((expected != 0).sum()) == kept


def test_ties_drops_entries_that_disagree_with_the_elected_sign():
    first = {"w": torch.tensor([3.0, 1.0, 0.1, 0.1])}
    second = {"w": torch.tensor([-1.0, 2.0, 0.1, 0.1])}
    merged = ties([first, second], retained_percentage=50.0)
    assert torch.equal(merged["w"], torch.tensor([3.0, 3.0, 0.0, 0.0]))


# ----------------------------------------------------------------- spectral
@pytest.mark.parametrize("method", [iso_c, iso_cts, tsv_m])
def test_spectral_methods_keep_shapes_and_average_other_tensors(method):
    task_vectors = random_task_vectors(3)
    merged = method(task_vectors, device="cpu")
    assert_same_shapes(merged, task_vectors[0])
    for name in ("layer.bias", "embedding"):
        assert torch.allclose(merged[name], torch.stack([tv[name] for tv in task_vectors]).mean(0))


def test_iso_c_flattens_the_spectrum_of_the_task_arithmetic_matrix():
    task_vectors = random_task_vectors(3)
    total = task_arithmetic(task_vectors)["layer.weight"]
    merged = iso_c(task_vectors, device="cpu")["layer.weight"]
    values = torch.linalg.svdvals(merged)
    assert torch.allclose(values, torch.full_like(values, torch.linalg.svdvals(total).mean()), atol=1e-5)


def test_iso_cts_has_a_flat_spectrum():
    values = torch.linalg.svdvals(iso_cts(random_task_vectors(2), device="cpu")["layer.weight"])
    assert torch.allclose(values, torch.full_like(values, values.mean()), atol=1e-5)


def test_tsv_m_with_one_task_reconstructs_the_task_matrix():
    (task_vector,) = random_task_vectors(1)
    merged = tsv_m([task_vector], device="cpu")
    assert torch.allclose(merged["layer.weight"], task_vector["layer.weight"], atol=1e-5)


def test_tsv_m_sums_tasks_that_live_in_orthogonal_subspaces():
    generator = torch.Generator().manual_seed(0)
    left = torch.linalg.qr(torch.randn(8, 8, generator=generator)).Q
    right = torch.linalg.qr(torch.randn(8, 8, generator=generator)).Q
    values = torch.tensor([4.0, 3.0, 2.0, 1.0])
    first = {"w": left[:, :4] @ torch.diag(values) @ right[:, :4].T}
    second = {"w": left[:, 4:] @ torch.diag(values) @ right[:, 4:].T}
    merged = tsv_m([first, second], device="cpu")["w"]
    assert torch.allclose(merged, first["w"] + second["w"], atol=1e-5)


def test_wudi_with_one_task_stays_at_the_task_vector():
    (task_vector,) = random_task_vectors(1)
    merged = wudi([task_vector], steps=5, device="cpu")
    assert all(torch.allclose(merged[name], task_vector[name]) for name in task_vector)


def test_wudi_reduces_its_interference_objective():
    task_vectors = [{"w": tv["layer.weight"]} for tv in random_task_vectors(3)]
    stacked = torch.stack([tv["w"] for tv in task_vectors])

    def interference(matrix: torch.Tensor) -> float:
        projected = (matrix.unsqueeze(0) - stacked) @ stacked.transpose(1, 2)
        return float((projected.square().sum(dim=(1, 2)) / stacked.square().sum(dim=(1, 2))).sum())

    merged = wudi(task_vectors, steps=50, learning_rate=1e-2, device="cpu")["w"]
    assert interference(merged) < interference(stacked.sum(0))


# ------------------------------------------------------------------ regmean
def linear_experts(count: int = 3, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return [
        {"fc.weight": torch.randn(5, 6, generator=generator), "fc.bias": torch.randn(5, generator=generator)}
        for _ in range(count)
    ]


def random_grams(count: int = 3, size: int = 6, seed: int = 1) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    inputs = [torch.randn(40, size, generator=generator) for _ in range(count)]
    return [{"fc": x.T @ x} for x in inputs]


def test_scale_off_diagonal_keeps_the_diagonal():
    gram = random_grams(1)[0]["fc"]
    scaled = scale_off_diagonal(gram, 0.25)
    assert torch.equal(scaled.diagonal(), gram.diagonal())
    assert torch.allclose(scaled - torch.diag(gram.diagonal()), 0.25 * (gram - torch.diag(gram.diagonal())))


def test_regmean_with_identity_grams_is_the_expert_mean():
    experts = linear_experts()
    merged = regmean(experts, [{"fc": torch.eye(6)}] * 3, ["fc"], off_diagonal_scale=1.0, device="cpu")
    for name in experts[0]:
        assert torch.allclose(merged[name], torch.stack([e[name] for e in experts]).mean(0), atol=1e-6)


def test_regmean_minimizes_the_layer_regression():
    experts, grams = linear_experts(), random_grams()
    merged = regmean(experts, grams, ["fc"], off_diagonal_scale=1.0, device="cpu")["fc.weight"]
    gradient = sum((merged - e["fc.weight"]) @ g["fc"] for e, g in zip(experts, grams, strict=True))
    assert float(gradient.abs().max()) < 1e-2


def test_regmean_rejects_singular_grams():
    with pytest.raises(torch.linalg.LinAlgError):
        regmean(
            linear_experts(), [{"fc": torch.zeros(6, 6)}] * 3, ["fc"], off_diagonal_scale=1.0, device="cpu"
        )


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(6)
        nn.init.normal_(self.norm.bias)  # keeps the inputs of fc full rank
        self.fc = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.fc(self.norm(x)))


def test_regmean_plus_plus_with_identical_experts_returns_the_expert():
    torch.manual_seed(0)
    expert = [TinyBlock().state_dict() for _ in range(2)]
    blocks = [TinyBlock(), TinyBlock()]
    inputs = [torch.randn(20, 6), torch.randn(20, 6)]
    regmean_plus_plus(blocks, [expert, expert], inputs, TinyBlock(), off_diagonal_scale=1.0, batch_size=8)
    for block, state in zip(blocks, expert, strict=True):
        for name, tensor in block.state_dict().items():
            assert torch.allclose(tensor, state[name], atol=1e-5)
    assert inputs[0].shape == (20, 6)


# ------------------------------------------------------------------- fisher
def test_diagonal_fisher_squares_the_batch_gradient():
    torch.manual_seed(0)
    model, inputs = nn.Linear(4, 3), torch.randn(6, 4)
    fisher = diagonal_fisher(model, (model(inputs[start : start + 3]) for start in (0, 3)))

    expected = torch.zeros_like(model.weight)
    for start in (0, 3):
        log_p = model(inputs[start : start + 3]).log_softmax(-1)
        (gradient,) = torch.autograd.grad((log_p.exp().detach().sqrt() * log_p).sum(), model.weight)
        expected += gradient**2
    assert torch.allclose(fisher["weight"], expected / 6)
    assert set(fisher) == {"weight", "bias"}


def test_fisher_merge_with_equal_fishers_is_the_mean():
    experts = linear_experts()
    fishers = [{name: torch.ones_like(tensor) for name, tensor in experts[0].items()}] * 3
    merged = fisher_merge(experts, fishers)
    for name in experts[0]:
        assert torch.allclose(merged[name], torch.stack([e[name] for e in experts]).mean(0), atol=1e-6)


def test_fisher_merge_follows_the_expert_with_the_larger_fisher():
    experts = linear_experts(2)
    fishers = [{"fc.weight": torch.full((5, 6), 1e3)}, {"fc.weight": torch.full((5, 6), 1e-9)}]
    merged = fisher_merge(experts, fishers, floor=0.0, normalize=False)
    assert torch.allclose(merged["fc.weight"], experts[0]["fc.weight"], atol=1e-4)
    assert torch.allclose(merged["fc.bias"], (experts[0]["fc.bias"] + experts[1]["fc.bias"]) / 2)


def test_mats_converges_to_the_regmean_solution():
    experts, grams = linear_experts(), random_grams()
    initial = {name: torch.zeros_like(tensor) for name, tensor in experts[0].items()}
    solved = mats(experts, grams, ["fc"], initial, iterations=200, device="cpu")
    reference = regmean(experts, grams, ["fc"], off_diagonal_scale=1.0, device="cpu")
    assert torch.allclose(solved["fc.weight"], reference["fc.weight"], atol=1e-3)
    assert torch.equal(solved["fc.bias"], initial["fc.bias"])


def test_mats_without_iterations_returns_the_initialization():
    experts, grams = linear_experts(), random_grams()
    solved = mats(experts, grams, ["fc"], experts[0], iterations=0, device="cpu")
    assert torch.allclose(solved["fc.weight"], experts[0]["fc.weight"])


# --------------------------------------------------------------- adamerging
def test_adamerging_starts_from_task_arithmetic_and_lowers_the_entropy():
    torch.manual_seed(0)
    pretrained = {"weight": torch.randn(3, 4), "bias": torch.zeros(3)}
    task_vectors = [{"weight": 0.3 * torch.randn(3, 4), "bias": 0.1 * torch.randn(3)} for _ in range(2)]
    inputs = torch.randn(32, 4)
    history = []

    def task_logits(task: int, state: dict) -> torch.Tensor:
        return inputs @ state["weight"].T + state["bias"]

    untouched = adamerging(pretrained, task_vectors, task_logits, steps=0, device="cpu")
    for name in pretrained:
        expected = pretrained[name] + 0.3 * (task_vectors[0][name] + task_vectors[1][name])
        assert torch.allclose(untouched[name], expected, atol=1e-6)

    trained = adamerging(
        pretrained,
        task_vectors,
        task_logits,
        steps=30,
        learning_rate=0.05,
        device="cpu",
        on_step=lambda *s: history.append(s[1]),
    )
    assert history[-1] < history[0]
    assert_same_shapes(trained, pretrained)


# ----------------------------------------------------------------- post-hoc
def test_lines_scales_linearly_with_depth():
    merged = {
        "blocks.0.w": torch.ones(2),
        "blocks.1.w": torch.ones(2),
        "blocks.2.w": torch.ones(2),
        "head": torch.ones(2),
    }

    def block_of(name: str) -> int | None:
        return int(name.split(".")[1]) if name.startswith("blocks") else None

    scaled = lines(merged, block_of, depth_ratio=1.0)
    assert [float(scaled[name][0]) for name in merged] == [1.0, 1.5, 2.0, 1.0]
    unchanged = lines(merged, block_of, depth_ratio=0.0)
    assert all(torch.equal(unchanged[name], merged[name]) for name in merged)


def test_svc_leaves_a_single_task_unchanged():
    (task_vector,) = random_task_vectors(1)
    calibrated = SingularValueCalibration([task_vector], task_vector, device="cpu").apply(
        scale=1.0, strength=0.1
    )
    for name in task_vector:
        assert torch.allclose(calibrated[name], task_vector[name], atol=1e-5)


def test_svc_undoes_the_over_accumulation_of_identical_tasks():
    (task_vector,) = random_task_vectors(1)
    task_vectors = [task_vector] * 3
    calibration = SingularValueCalibration(task_vectors, task_arithmetic(task_vectors), device="cpu")
    calibrated = calibration.apply(scale=1.0, strength=0.1)
    assert torch.allclose(calibrated["layer.weight"], task_vector["layer.weight"], atol=1e-5)
    assert torch.allclose(calibrated["layer.bias"], 3 * task_vector["layer.bias"])
    half = calibration.apply(scale=0.5, strength=0.1)  # the scale is applied before the calibration
    assert torch.allclose(half["layer.weight"], task_vector["layer.weight"], atol=1e-5)


# ------------------------------------------------------------------ reforge
def reforge_inputs():
    experts, grams = linear_experts(), random_grams()
    experts = [{f"resblocks.0.{name}": tensor for name, tensor in expert.items()} for expert in experts]
    grams = [{"resblocks.0.fc": gram["fc"]} for gram in grams]
    pretrained = {name: torch.zeros_like(tensor) for name, tensor in experts[0].items()}
    layers = ["resblocks.0.fc"]
    return pretrained, experts, grams, layers, depth_groups(layers, groups=1)


def test_anchored_regression_interpolates_between_regmean_and_the_anchor():
    pretrained, experts, grams, layers, group_of = reforge_inputs()
    anchor = {"resblocks.0.fc.weight": torch.randn(5, 6, generator=torch.Generator().manual_seed(3))}

    def solve(log_ridge: float, output_scale: float = 1.0) -> torch.Tensor:
        arguments = (
            pretrained,
            experts,
            grams,
            layers,
            anchor,
            [log_ridge],
            [1.0],
            [output_scale],
            group_of,
            "cpu",
        )
        return anchored_regression(*arguments)["resblocks.0.fc.weight"]

    reference = regmean(experts, grams, layers, off_diagonal_scale=1.0, device="cpu")["resblocks.0.fc.weight"]
    assert torch.allclose(solve(-9.0), reference, atol=1e-3)
    assert torch.allclose(solve(9.0), anchor["resblocks.0.fc.weight"], atol=1e-3)
    assert torch.allclose(solve(9.0, output_scale=0.5), 0.5 * anchor["resblocks.0.fc.weight"], atol=1e-3)


def test_depth_groups_split_the_blocks_evenly():
    layers = [f"resblocks.{index}.fc" for index in range(12)]
    group_of = depth_groups(layers, groups=3)
    assert [group_of(layer) for layer in layers] == [0] * 4 + [1] * 4 + [2] * 4


@pytest.mark.filterwarnings("ignore")
def test_gaussian_process_optimizer_finds_a_smooth_maximum():
    pytest.importorskip("sklearn")
    optimizer = GaussianProcessOptimizer([(-2.0, 2.0), (-2.0, 2.0)], seed=0, initial=8, candidates=256)
    for _ in range(20):
        point = optimizer.ask()
        optimizer.tell(point, -((point[0] - 1.0) ** 2) - (point[1] + 0.5) ** 2)
    point, score = optimizer.best()
    assert score > -0.5 and all(-2.0 <= value <= 2.0 for value in point)


# ----------------------------------------------------------------- registry
def test_every_baseline_has_settings_and_grids_are_valid():
    settings = yaml.safe_load((CONFIGS / "clip.yaml").read_text(encoding="utf-8"))
    assert set(settings) == set(BASELINES)
    for method, entry in settings.items():
        grid = (entry or {}).get("grid", {})
        assert all(len(values) == len(set(values)) > 0 for values in grid.values()), method


def test_selected_hyperparameters_lie_on_the_grids():
    settings = yaml.safe_load((CONFIGS / "clip.yaml").read_text(encoding="utf-8"))
    selected = yaml.safe_load((CONFIGS / "clip_selected.yaml").read_text(encoding="utf-8"))
    for key, per_architecture in selected.items():
        method = key.split("+")[0]
        assert method in BASELINES and BASELINES[method].posthoc == ("+" in key)
        for architecture, per_count in per_architecture.items():
            for task_count, values in per_count.items():
                grid = settings[method].get("grid", {})
                for name, value in values.items():
                    if name in grid:
                        assert value in grid[name], (key, architecture, task_count, name)


# ------------------------------------------------------------- CLIP driver
def test_settings_are_the_same_for_every_backbone_and_accept_overrides():
    assert clip_baselines.baseline_settings("ta", "ViT-B-32", 8) == clip_baselines.baseline_settings(
        "ta", "ViT-L-14", 20
    )
    assert len(clip_baselines.baseline_settings("ta", "ViT-L-14", 8)["grid"]["scale"]) == 12
    assert (
        clip_baselines.baseline_settings("ties", "ViT-B-16", 20, {"retained_percentage": 10.0})[
            "retained_percentage"
        ]
        == 10.0
    )


def test_grid_points_are_ordered_with_the_last_hyperparameter_fastest():
    points = clip_baselines._grid({"grid": {"strength": [0.1, 0.2], "scale": [1.0, 2.0]}})
    assert points[:3] == [
        {"strength": 0.1, "scale": 1.0},
        {"strength": 0.1, "scale": 2.0},
        {"strength": 0.2, "scale": 1.0},
    ]
    assert clip_baselines._grid({}) == [{}]


def test_selected_hyperparameterss_and_output_directories():
    assert clip_baselines.selected_hyperparameters("ta", "ViT-B-32", 8) == {"scale": 0.25}
    assert clip_baselines.selected_hyperparameters("lines", "ViT-B-32", 8, init="wa") == {"scale": 1.35}
    assert clip_baselines.selected_hyperparameters("fisher", "ViT-B-32", 8) is None
    directory = clip_baselines.baseline_directory("runs", "ViT-B-32", 8, "lines", "ta")
    assert directory == Path("runs/clip/ViT-B-32-8tasks/baselines/lines+ta")
    assert (
        clip_baselines.baseline_directory("runs", "ViT-L-14", 20, "svc", "some/merged.pt").name
        == "svc+merged"
    )


def test_run_baseline_checks_the_method_and_its_initialization():
    for method, init in (("unknown", None), ("lines", None), ("ta", "wa")):
        with pytest.raises(ValueError):
            clip_baselines.run_baseline(method, "ViT-B-32", 8, assets=None, device="cpu", init=init)
