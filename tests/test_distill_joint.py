"""The search protocol of the distillation baseline and the data pipeline of joint training."""

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("open_clip")

from if_merge.clip.assets import ClipAssets  # noqa: E402
from if_merge.clip.distill import (  # noqa: E402
    SearchSettings,
    Trial,
    budget_split,
    evaluation_interval,
    next_learning_rate,
    search_learning_rates,
    teacher_student_kl,
)
from if_merge.clip.joint import (  # noqa: E402
    BalancedBatches,
    TrainingImages,
    convert_released,
    learning_rate_at,
)


def test_budget_split_keeps_the_default_holdout():
    assert budget_split(512) == (list(range(256)), list(range(256, 512)))
    assert budget_split(128) == (list(range(64)), list(range(64, 128)))
    statistics, holdout = budget_split(2048)
    assert holdout == list(range(256, 512))
    assert statistics == list(range(256)) + list(range(512, 2048))
    with pytest.raises(ValueError):
        budget_split(127)


def test_evaluation_interval_is_two_passes_within_bounds():
    assert [evaluation_interval(n, 16) for n in (16, 64, 256, 1792)] == [5, 8, 32, 100]


def test_learning_rate_grid_neighbors():
    for rate, direction, expected in ((3e-6, -1, 1e-6), (1e-6, -1, 3e-7), (1e-5, 1, 3e-5), (3e-5, 1, 1e-4)):
        assert next_learning_rate(rate, direction) == pytest.approx(expected)


def test_teacher_student_kl_matches_the_definition():
    torch.manual_seed(0)
    teacher, student = torch.randn(5, 7), torch.randn(5, 7)
    expected = F.kl_div(student.log_softmax(-1), teacher.softmax(-1), reduction="batchmean")
    assert teacher_student_kl(teacher, student) == pytest.approx(float(expected), abs=1e-6)
    assert float(teacher_student_kl(teacher, teacher)) == pytest.approx(0.0, abs=1e-7)


def fake_training(best_kl, improving_at_cap=()):
    """A ``train`` function whose holdout KL depends only on the learning rate."""
    calls = []

    def train(rate: float, max_steps: int) -> Trial:
        calls.append((rate, max_steps))
        capped = any(math.isclose(rate, r) for r in improving_at_cap) and max_steps == 6000
        return Trial(
            rate,
            best_kl(rate),
            100,
            {},
            [],
            "max_steps" if capped else "patience",
            200,
            max_steps,
            capped,
            0.0,
        )

    return train, calls


def test_search_stops_at_an_interior_learning_rate():
    train, calls = fake_training(lambda rate: abs(math.log10(rate) + 5))  # best at 1e-5
    trials, selected = search_learning_rates(train, SearchSettings())
    assert selected == 1e-5
    assert [rate for rate, _ in calls] == [3e-6, 1e-5, 3e-5]


def test_search_extends_the_grid_beyond_an_edge():
    train, calls = fake_training(lambda rate: abs(math.log10(rate) + 6))  # best at 1e-6
    trials, selected = search_learning_rates(train, SearchSettings())
    assert selected == pytest.approx(1e-6)
    assert len(trials) == 5  # 1e-6 and 3e-7 were added below the grid
    # the extension is bounded: a minimum far below the grid is not reached
    train, calls = fake_training(lambda rate: rate)
    trials, selected = search_learning_rates(train, SearchSettings(extensions=1))
    assert selected == pytest.approx(1e-6) and len(trials) == 4


def test_search_doubles_the_steps_of_a_capped_selection():
    train, calls = fake_training(lambda rate: abs(math.log10(rate) + 5), improving_at_cap=(1e-5,))
    trials, selected = search_learning_rates(train, SearchSettings())
    assert calls[-1] == (1e-5, 12000)
    assert trials[selected].max_steps == 12000 and not trials[selected].cap_limited


def test_balanced_batches_draw_evenly_without_repetition():
    lengths = [10, 4, 25]
    sampler = BalancedBatches(lengths, batch_size=7, steps=6, seed=0)
    batches = list(sampler)
    assert batches == list(BalancedBatches(lengths, batch_size=7, steps=6, seed=0))
    assert batches != list(BalancedBatches(lengths, batch_size=7, steps=6, seed=1))
    bounds = torch.tensor([10, 14, 39])
    for batch in batches:
        tasks = torch.bucketize(torch.tensor(batch), bounds, right=True)
        assert torch.bincount(tasks, minlength=3).tolist() == [3, 2, 2]
    first_task = [index for batch in batches[:3] for index in batch if index < 10]
    assert len(first_task) == 9 and len(set(first_task)) == 9  # one pass before reshuffling


def test_learning_rate_schedule():
    assert learning_rate_at(0, 1e-5, 3000) == pytest.approx(1e-5 / 500)
    assert learning_rate_at(499, 1e-5, 3000) == pytest.approx(1e-5)
    assert learning_rate_at(1750, 1e-5, 3000) == pytest.approx(0.5e-5)
    assert learning_rate_at(3000, 1e-5, 3000) == pytest.approx(0.0, abs=1e-12)


def test_training_images_exclude_the_validation_split(tmp_path):
    assets = ClipAssets("ViT-B-32", tmp_path)
    sizes, validation = {"DTD": 6, "MNIST": 5}, {"DTD": [1, 4], "MNIST": [0]}
    for task, size in sizes.items():
        images = torch.arange(size, dtype=torch.uint8).reshape(-1, 1, 1, 1).expand(size, 3, 224, 224).clone()
        path = assets.image_path(task, "train")
        path.parent.mkdir(parents=True)
        torch.save(
            {"x": images, "y": torch.arange(size) + 100, "val_idx": torch.tensor(validation[task])}, path
        )
    data = TrainingImages(assets, ["DTD", "MNIST"])
    assert data.lengths == [4, 4] and len(data) == 8
    images, labels, task_ids = data[[7, 0, 5, 2]]
    assert task_ids.tolist() == [1, 0, 1, 0]
    assert labels.tolist() == [104, 100, 102, 103]  # position 2 of DTD is image 3: 1 and 4 are validation
    assert images[:, 0, 0, 0].tolist() == [4, 0, 2, 3]


def test_convert_released_checkpoint_layout():
    torch.manual_seed(0)
    width, blocks = 8, 2
    pretrained = {
        "class_embedding": torch.zeros(width),
        "conv1.weight": torch.zeros(width, 3, 4, 4),
        "positional_embedding": torch.zeros(5, width),
        "proj": torch.randn(width, 6),
        **{
            f"ln_{name}.{kind}": torch.zeros(width) for name in ("pre", "post") for kind in ("weight", "bias")
        },
    }
    released = {
        "vision_model.embeddings.class_embedding": torch.randn(width),
        "vision_model.embeddings.patch_embedding.weight": torch.randn(width, 3, 4, 4),
        "vision_model.embeddings.position_embedding.weight": torch.randn(5, width),
        **{
            f"vision_model.{name}.{kind}": torch.randn(width)
            for name in ("pre_layrnorm", "post_layernorm")
            for kind in ("weight", "bias")
        },
    }
    for block in range(blocks):
        ours, theirs = f"transformer.resblocks.{block}.", f"vision_model.encoder.layers.{block}."
        pretrained[f"{ours}attn.in_proj_weight"] = torch.zeros(3 * width, width)
        pretrained[f"{ours}attn.in_proj_bias"] = torch.zeros(3 * width)
        for name in "qkv":
            released[f"{theirs}self_attn.{name}_proj.weight"] = torch.randn(width, width)
            released[f"{theirs}self_attn.{name}_proj.bias"] = torch.randn(width)
        shapes = {
            "attn.out_proj": (width, width),
            "mlp.c_fc": (4 * width, width),
            "mlp.c_proj": (width, 4 * width),
        }
        names = {"attn.out_proj": "self_attn.out_proj", "mlp.c_fc": "mlp.fc1", "mlp.c_proj": "mlp.fc2"}
        for name, shape in shapes.items():
            pretrained[f"{ours}{name}.weight"], pretrained[f"{ours}{name}.bias"] = (
                torch.zeros(shape),
                torch.zeros(shape[0]),
            )
            released[f"{theirs}{names[name]}.weight"] = torch.randn(shape)
            released[f"{theirs}{names[name]}.bias"] = torch.randn(shape[0])
        for ours_name, their_name in (("ln_1", "layer_norm1"), ("ln_2", "layer_norm2")):
            for kind in ("weight", "bias"):
                pretrained[f"{ours}{ours_name}.{kind}"] = torch.zeros(width)
                released[f"{theirs}{their_name}.{kind}"] = torch.randn(width)

    state = convert_released(released, pretrained)
    assert state.keys() == pretrained.keys()
    assert torch.equal(state["proj"], pretrained["proj"])  # the released model has no projection
    key = released["vision_model.encoder.layers.1.self_attn.k_proj.weight"]
    assert torch.equal(state["transformer.resblocks.1.attn.in_proj_weight"][width : 2 * width], key)
    assert torch.equal(
        state["transformer.resblocks.0.mlp.c_fc.bias"], released["vision_model.encoder.layers.0.mlp.fc1.bias"]
    )
    del released["vision_model.encoder.layers.1.layer_norm1.weight"]  # a shallower model
    with pytest.raises(ValueError):
        convert_released(released, pretrained)
