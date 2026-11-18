"""The architecture-agnostic core: statistics, the layerwise fit and the IF-Merge loop."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from if_merge import IFMergeConfig, if_merge
from if_merge.core.solver import condition_moment, conjugate_gradient, solve_layer
from if_merge.core.statistics import (
    SecondMoment,
    StatisticsCollector,
    TaskStatistics,
    adjust_feedback,
    combine_statistics,
    train_only,
)


def spd(dimension: int, seed: int) -> torch.Tensor:
    x = torch.randn(
        4 * dimension, dimension, generator=torch.Generator().manual_seed(seed), dtype=torch.float64
    )
    return x.T @ x / len(x)


def test_conjugate_gradient_solves_spd_system():
    matrix, target = spd(6, 0), torch.randn(6, 3, dtype=torch.float64)
    solution = conjugate_gradient(lambda x: matrix @ x, target, torch.zeros_like(target), 100, 1e-12)
    assert torch.allclose(matrix @ solution, target, atol=1e-8)


def test_second_moment_and_shrinkage():
    moment, samples = SecondMoment(), torch.randn(50, 7, 4)
    moment.update(samples[:20])
    moment.update(samples[20:])
    flat = samples.reshape(-1, 4)
    assert torch.allclose(moment.mean(), flat.T @ flat / len(flat), atol=1e-5)
    assert 0.0 <= moment.ledoit_wolf_shrinkage() <= 1.0


def test_condition_moment():
    moment = spd(5, 1).float()
    assert torch.allclose(condition_moment(moment).diagonal().mean(), torch.tensor(1.0))
    assert torch.equal(condition_moment(moment, 1.0), torch.eye(5))
    assert not condition_moment(torch.zeros(5, 5), 0.5).any()  # no feedback stays no feedback


def test_solve_layer_matches_closed_form():
    """With two-sided metrics, vec(X) solves sum_t (G_t kron A_t) vec(X) = sum_t vec(A_t W_t G_t)."""
    torch.manual_seed(0)
    outputs, inputs, tasks = 3, 4, 3
    experts = [torch.randn(outputs, inputs, dtype=torch.float64) for _ in range(tasks)]
    A = [spd(inputs, 10 + t) for t in range(tasks)]
    G = [spd(outputs, 20 + t) for t in range(tasks)]
    solver = IFMergeConfig.from_mapping(
        {"solver": {"max_iterations": 500, "relative_tolerance": 1e-13}}
    ).solver
    solved = solve_layer(experts, A, G, torch.zeros(outputs, inputs, dtype=torch.float64), solver)
    operator = sum(torch.kron(a, g) for a, g in zip(A, G, strict=True))  # acts on row-major vec of X^T
    target = sum((a @ w.T @ g).reshape(-1) for a, w, g in zip(A, experts, G, strict=True))
    reference = torch.linalg.solve(operator, target).reshape(inputs, outputs).T
    assert torch.allclose(solved, reference, atol=1e-7)


def test_activation_space_fit_is_regmean():
    experts = [torch.randn(3, 4, dtype=torch.float64) for _ in range(2)]
    A = [spd(4, 1), spd(4, 2)]
    solver = IFMergeConfig.from_mapping(
        {"solver": {"max_iterations": 500, "relative_tolerance": 1e-13}}
    ).solver
    solved = solve_layer(experts, A, None, torch.zeros(3, 4, dtype=torch.float64), solver)
    reference = torch.linalg.solve(A[0] + A[1], A[0] @ experts[0].T + A[1] @ experts[1].T).T
    assert torch.allclose(solved, reference, atol=1e-7)


def test_identical_experts_are_a_fixed_point_and_no_feedback_keeps_the_layer():
    weight = torch.randn(3, 4, dtype=torch.float64)
    solver = IFMergeConfig().solver
    solved = solve_layer(
        [weight, weight], [spd(4, 1), spd(4, 2)], [spd(3, 3), spd(3, 4)], weight.clone(), solver
    )
    assert torch.allclose(solved, weight)
    zero = torch.zeros(3, 3, dtype=torch.float64)
    assert solve_layer([weight, weight], [spd(4, 1), spd(4, 2)], [zero, zero], weight, solver) is None


def test_combine_statistics_keeps_first_round_factors():
    first, fresh = [TaskStatistics({"l": 1}, {"l": 2})], [TaskStatistics({"l": 3}, {"l": 4})]
    assert combine_statistics(fresh, None, "a") is fresh
    assert combine_statistics(fresh, first, "none") is first
    assert (combine_statistics(fresh, first, "a")[0].A, combine_statistics(fresh, first, "a")[0].G) == (
        {"l": 3},
        {"l": 2},
    )
    assert (combine_statistics(fresh, first, "g")[0].A, combine_statistics(fresh, first, "g")[0].G) == (
        {"l": 1},
        {"l": 4},
    )


class ToyBackend:
    """Two-layer MLP students and experts on random inputs; teachers are the experts' logits."""

    activation_only = frozenset()

    def __init__(self, tasks: int = 3, width: int = 16, classes: int = 5, samples: int = 64) -> None:
        generator = torch.Generator().manual_seed(0)
        self.device = "cpu"

        def mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(8, width, bias=False), nn.Tanh(), nn.Linear(width, classes, bias=False)
            )

        base = mlp()
        self.experts = []
        for _ in range(tasks):
            expert = mlp()
            with torch.no_grad():
                for target, source in zip(expert.parameters(), base.parameters(), strict=True):
                    target.copy_(source + 0.5 * torch.randn(source.shape, generator=generator))
            self.experts.append(expert)
        self.model = mlp()
        with torch.no_grad():
            for index, parameter in enumerate(self.model.parameters()):
                parameter.copy_(torch.stack([list(e.parameters())[index] for e in self.experts]).mean(0))
        self.layers = {name: m for name, m in self.model.named_modules() if isinstance(m, nn.Linear)}
        self.blocks = {index: [name] for index, name in enumerate(self.layers)}
        self.inputs = [torch.randn(2 * samples, 8, generator=generator) for _ in range(tasks)]
        with torch.no_grad():
            self.teachers = [expert(x) for expert, x in zip(self.experts, self.inputs, strict=True)]
        self.samples = samples

    def expert_weights(self, layer):
        return [dict(expert.named_modules())[layer].weight.detach() for expert in self.experts]

    def state(self):
        return {name: value.detach().clone() for name, value in self.model.state_dict().items()}

    @torch.no_grad()
    def holdout_kl(self) -> float:
        total = 0.0
        for x, teacher in zip(self.inputs, self.teachers, strict=True):
            logits = self.model(x[self.samples :])
            total += float(
                F.kl_div(logits.log_softmax(-1), teacher[self.samples :].softmax(-1), reduction="batchmean")
            )
        return total / len(self.inputs)

    def collect_statistics(self, config, round_index):
        train_only(self.model, self.layers.values())
        statistics = []
        for x, teacher in zip(self.inputs, self.teachers, strict=True):
            with StatisticsCollector(self.layers, feedback=config.g.estimator != "identity") as collector:
                logits = self.model(x[: self.samples])
                if config.g.estimator != "identity":
                    F.kl_div(
                        logits.log_softmax(-1), teacher[: self.samples].softmax(-1), reduction="sum"
                    ).backward()
            statistics.append(collector.finish())
        train_only(self.model, ())
        return statistics


@pytest.mark.parametrize(
    "overrides", [[], ["selection.scope=per_block"], ["g.estimator=identity"], ["iteration.reestimate=none"]]
)
def test_if_merge_never_increases_holdout_kl(overrides):
    backend, config = ToyBackend(), IFMergeConfig().with_overrides(overrides)
    seen = []
    result = if_merge(backend, config, seen.append)
    kls = [record.holdout_kl for record in result.rounds]
    assert seen == result.rounds and result.rounds[0].round == 0
    assert all(later <= earlier + 1e-9 for earlier, later in zip(kls, kls[1:], strict=False))
    assert kls[result.selected_round] == min(kls) < kls[0]
    backend.model.load_state_dict(result.state)
    assert backend.holdout_kl() == pytest.approx(min(kls), abs=1e-6)


def test_recipe_overrides_and_validation():
    config = IFMergeConfig().with_overrides(
        ["iteration.rounds=8", "g.shrinkage=0.3", "selection.steps=[0, 1]"]
    )
    assert (config.iteration.rounds, config.g.shrinkage, config.selection.steps) == (8, 0.3, (0.0, 1.0))
    for bad in ("g.estimator=nope", "selection.steps=[0.5, 1]", "iteration.unknown=1", "rounds=3"):
        with pytest.raises(ValueError):
            IFMergeConfig().with_overrides([bad])
    assert IFMergeConfig.from_mapping(config.to_dict()) == config


def test_adjust_feedback_permutes_and_shares():
    def stats():
        return [
            TaskStatistics({"l": torch.eye(2)}, {"l": torch.diag(torch.tensor([s, 3.0 * s]))}, {"l": 0.1 * s})
            for s in (1.0, 2.0, 4.0)
        ]

    permuted = adjust_feedback(stats(), permute_tasks=True)
    assert [float(task.G["l"][0, 0]) for task in permuted] == [2.0, 4.0, 1.0]
    assert [task.G_shrinkage["l"] for task in permuted] == pytest.approx([0.2, 0.4, 0.1])
    shared = adjust_feedback(stats(), heterogeneity=0.0)
    assert all(torch.allclose(task.G["l"], shared[0].G["l"]) for task in shared)
    assert torch.allclose(shared[0].G["l"].diagonal().mean(), torch.tensor(1.0))  # scale-normalized
    assert shared[0].G_shrinkage["l"] == pytest.approx((0.1 + 0.2 + 0.4) / 3)
    untouched = adjust_feedback(stats())
    assert float(untouched[2].G["l"][1, 1]) == 12.0


def test_forced_steps_replace_selection():
    backend, config = (
        ToyBackend(),
        IFMergeConfig().with_overrides(
            ["selection.forced_steps=[1.0, 0.5]", "selection.scope=per_block", "iteration.rounds=3"]
        ),
    )
    rounds = if_merge(backend, config).rounds
    assert [set(r.block_steps.values()) for r in rounds[1:3]] == [{1.0}, {0.5}]
    assert all(r.statistics_seconds is not None for r in rounds[1:])


def test_zero_warm_start_and_output_activation_run():
    for override in ("solver.warm_start=zero", "g.permute_tasks=true", "g.heterogeneity=0.5"):
        assert if_merge(ToyBackend(), IFMergeConfig().with_overrides([override])).rounds[-1].holdout_kl > 0
    layer = nn.Linear(3, 2, bias=False)
    with StatisticsCollector({"l": layer}, output_activations=True) as collector:
        output = layer(torch.randn(10, 3))
    assert torch.allclose(collector.finish().G["l"], output.T @ output / 10, atol=1e-6)
