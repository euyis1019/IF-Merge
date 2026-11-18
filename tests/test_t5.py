"""The Flan-T5 backend, baselines and command line, on tiny randomly initialized models."""

import argparse
import copy
import json

import pytest
import torch
from safetensors.torch import save_file

from if_merge import IFMergeConfig, if_merge

transformers = pytest.importorskip("transformers")

from if_merge.core.statistics import StatisticsCollector  # noqa: E402
from if_merge.t5 import baselines, run  # noqa: E402
from if_merge.t5.backend import T5Backend, Teacher, forward, valid_rows  # noqa: E402
from if_merge.t5.data import (  # noqa: E402
    CALIBRATION_POOL,
    RESERVED_FOR_CALIBRATION,
    SELECTION_EXAMPLES,
    TASKS,
    T5Assets,
    calibration_index,
    prompt_and_answer,
    selection_index,
)
from if_merge.t5.evaluate import score  # noqa: E402
from if_merge.t5.model import (  # noqa: E402
    EMBEDDING,
    EMBEDDING_ALIASES,
    HEAD,
    load_model,
    load_state,
    merged_layers,
    model_state,
    save_state,
)
from if_merge.t5.prepare import merge_adapter  # noqa: E402

VOCABULARY, PROMPTS, PROMPT_LENGTH = 48, 12, 9
RECIPE = run.RECIPE


def tiny_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    config = transformers.T5Config(
        vocab_size=VOCABULARY, d_model=16, d_kv=4, d_ff=24, num_layers=2, num_decoder_layers=2, num_heads=4,
        feed_forward_proj="gated-gelu", tie_word_embeddings=False, decoder_start_token_id=0, dropout_rate=0.0,
    )  # fmt: skip
    return transformers.T5ForConditionalGeneration(config).eval()


def tiny_assets(size: str, root) -> T5Assets:
    """A data root with a tiny pretrained model, eight perturbed experts and cached prompts."""
    assets = T5Assets(size, root)
    pretrained = tiny_model(0)
    pretrained.save_pretrained(assets.model_dir())
    generator = torch.Generator().manual_seed(1)
    for task in TASKS:
        expert = copy.deepcopy(pretrained)
        with torch.no_grad():
            for parameter in expert.parameters():
                parameter += 0.05 * torch.randn(parameter.shape, generator=generator)
        expert.save_pretrained(assets.model_dir(task))
        lengths = torch.randint(3, PROMPT_LENGTH + 1, (PROMPTS,), generator=generator)
        mask = (torch.arange(PROMPT_LENGTH)[None, :] < lengths[:, None]).to(torch.int8)
        ids = torch.randint(2, VOCABULARY, (PROMPTS, PROMPT_LENGTH), generator=generator) * mask
        path = assets.cache_path(task, f"calibration_n{PROMPTS}_seed0")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"input_ids": ids.to(torch.int32), "attention_mask": mask}, path)
    return assets


@pytest.fixture(scope="module", params=["base", "large"])
def assets(request, tmp_path_factory) -> T5Assets:
    return tiny_assets(request.param, tmp_path_factory.mktemp(request.param))


def tiny_config(*overrides: str) -> IFMergeConfig:
    samples = [f"calibration.samples={PROMPTS // 2}", f"calibration.holdout={PROMPTS // 2}"]
    return IFMergeConfig.load(RECIPE, [*samples, "iteration.rounds=1", *overrides])


@pytest.fixture(scope="module")
def backend(assets) -> T5Backend:
    initial = model_state(load_model(assets))
    return T5Backend(assets, initial, tiny_config(), device="cpu", batch_size=4, tasks=TASKS[:3])


def test_assets_resolve_each_size(assets):
    assert assets.model_dir().parts[-3:] == ("models", assets.size, "pretrained")
    assert assets.model_dir("rte").parts[-3:] == ("models", assets.size, "rte")
    assert assets.adapter_dir("rte").parts[-4:] == ("models", assets.size, "adapters", "rte")
    assert assets.cache_path("rte", "validation").parent.name == assets.size


def test_teacher_marks_positions_up_to_end_of_sequence(backend):
    teacher = backend.calibration[TASKS[0]]
    assert teacher.logits.dtype == torch.float16 and teacher.logits.shape[:2] == teacher.valid.shape
    assert teacher.logits.shape[-1] == VOCABULARY
    for answer, valid in zip(teacher.decoder_input_ids, teacher.valid, strict=True):
        length = int(valid.sum())
        assert length >= 1 and bool(valid[:length].all()) and not bool(valid[length:].any())
        assert 1 not in answer[1:length].tolist()  # no end-of-sequence before the last valid position


def test_statistics_cover_every_layer(backend):
    statistics = backend.collect_statistics(tiny_config(), round_index=1)
    assert len(statistics) == 3
    for task in statistics:
        assert set(task.A) == set(backend.layers)
        assert set(task.G) == set(backend.layers) - {HEAD}  # the vocabulary head is fitted with G = I
        for name, layer in backend.layers.items():
            assert task.A[name].shape == (layer.in_features, layer.in_features)
            if name != HEAD:
                assert task.G[name].shape == (layer.out_features, layer.out_features)
                assert 0.0 <= task.G_shrinkage[name] <= 1.0


def test_collector_counts_valid_positions_only(backend):
    masks = {}
    batch = next(backend.calibration[TASKS[0]].batches(4, "cpu"))
    assert int(batch.attention_mask.sum()) < batch.attention_mask.numel()  # the batch does contain padding
    layers = merged_layers(backend.model)
    with StatisticsCollector(layers, feedback=False, rows=valid_rows(masks)) as collector, torch.no_grad():
        masks["encoder"], masks["decoder"] = batch.attention_mask.bool(), batch.valid.bool()
        forward(backend.model, batch)
    prompt_positions, answer_positions = int(batch.attention_mask.sum()), int(batch.valid.sum())
    counts = {name: moment.count for name, moment in collector.inputs.items()}
    assert counts["encoder.block.0.layer.0.SelfAttention.q"] == prompt_positions
    assert counts["decoder.block.0.layer.1.EncDecAttention.k"] == prompt_positions  # reads the encoder
    assert counts["decoder.block.0.layer.1.EncDecAttention.q"] == answer_positions
    assert counts["decoder.block.1.layer.2.DenseReluDense.wo"] == answer_positions
    assert counts[HEAD] == answer_positions


def test_statistics_ignore_padding(backend):
    """Extra padding columns change neither A nor G."""
    task = TASKS[0]
    reference = backend.collect_statistics(tiny_config(), 1)[0]
    original = backend.calibration[task]
    pad = torch.nn.functional.pad
    backend.calibration[task] = Teacher(
        pad(original.input_ids, (0, 4)),
        pad(original.attention_mask, (0, 4)),
        pad(original.decoder_input_ids, (0, 3)),
        pad(original.valid, (0, 3)),
        pad(original.logits, (0, 0, 0, 3)),
    )
    try:
        padded = backend.collect_statistics(tiny_config(), 1)[0]
    finally:
        backend.calibration[task] = original
    for name in reference.A:
        assert torch.allclose(reference.A[name], padded.A[name], atol=1e-5), name
    for name in reference.G:
        scale = reference.G[name].abs().max()
        assert torch.allclose(reference.G[name], padded.G[name], atol=1e-4 * float(scale)), name


@pytest.mark.parametrize("estimator", ["kl", "mc_fisher", "identity"])
def test_other_estimators(backend, estimator):
    statistics = backend.collect_statistics(tiny_config(f"g.estimator={estimator}"), 1)[0]
    assert set(statistics.A) == set(backend.layers)
    assert bool(statistics.G) == (estimator != "identity")


def test_unsupported_settings_are_rejected(assets):
    initial = model_state(load_model(assets))
    with pytest.raises(ValueError, match="g.estimator"):
        T5Backend(assets, initial, tiny_config("g.estimator=exact_fisher"), device="cpu")
    with pytest.raises(ValueError, match="selection.scope"):
        T5Backend(assets, initial, tiny_config("selection.scope=per_block"), device="cpu")


@pytest.mark.parametrize("freeze_head", [False, True])
def test_one_round_of_if_merge(assets, freeze_head):
    initial = model_state(load_model(assets))
    config = tiny_config()
    backend = T5Backend(
        assets, initial, config, "cpu", batch_size=4, freeze_head=freeze_head, tasks=TASKS[:3]
    )
    assert (HEAD in backend.layers) != freeze_head and list(backend.blocks) == ["model"]
    result = if_merge(backend, config)

    before, after = result.rounds
    assert after.holdout_kl <= before.holdout_kl and after.global_step > 0
    assert set(after.block_steps) == {"model"}
    # the returned state is the selected model, embeddings untouched and still tied
    assert backend.holdout_kl() == pytest.approx(result.rounds[result.selected_round].holdout_kl, rel=1e-5)
    assert backend.model.encoder.embed_tokens is backend.model.shared is backend.model.decoder.embed_tokens
    for alias in EMBEDDING_ALIASES:
        assert result.state[alias] is result.state[EMBEDDING]
    assert torch.equal(result.state[EMBEDDING], initial[EMBEDDING])
    changed = {key for key in initial if not torch.equal(initial[key], result.state[key])}
    assert changed <= {f"{name}.weight" for name in backend.layers} and changed
    assert (f"{HEAD}.weight" in changed) != freeze_head


def test_state_round_trip(assets, tmp_path):
    state = model_state(load_model(assets, "cola"))
    save_state(state, tmp_path / "merged.safetensors")
    loaded = load_state(tmp_path / "merged.safetensors")
    assert set(loaded) == set(state) and all(torch.equal(loaded[key], state[key]) for key in state)
    assert loaded[EMBEDDING_ALIASES[0]] is loaded[EMBEDDING]


def test_selection_split_is_disjoint_from_calibration():
    examples = 2490  # the smallest training split (RTE)
    for size, reserved in RESERVED_FOR_CALIBRATION.items():
        calibration = {
            int(i) for seed in reserved for i in calibration_index(examples, CALIBRATION_POOL, seed).tolist()
        }
        for seed in reserved:
            selection = selection_index(examples, size, seed).tolist()
            assert len(set(selection)) == SELECTION_EXAMPLES and not calibration & set(selection)


def test_prompts_and_scores():
    prompt, answer = prompt_and_answer("stsb", {"sentence1": "a", "sentence2": "b", "label": 3.8})
    assert "'a' and 'b'" in prompt and answer == "3.8"
    assert prompt_and_answer("mnli", {"premise": "p", "hypothesis": "h", "label": 2})[1] == "contradiction"
    assert prompt_and_answer("rte", {"sentence1": "p", "sentence2": "h", "label": -1})[1] == ""
    assert score("cola", ["acceptable", "x"], ["acceptable", "unacceptable"]) == 50.0
    assert score("stsb", ["1.0", "2.0", "oops"], ["1.0", "2.0", "3.0"]) == pytest.approx(-50.0)
    assert score("stsb", ["1.0", "1.0"], ["1.0", "2.0"]) == 0.0  # constant predictions


def test_merge_adapter(tmp_path):
    model, layer = tiny_model(0), "encoder.block.0.layer.0.SelfAttention.q"
    before = model.get_submodule(layer).weight.clone()
    down, up = torch.randn(2, 16), torch.randn(16, 2)
    prefix = f"base_model.model.{layer}"
    save_file(
        {f"{prefix}.lora_A.weight": down, f"{prefix}.lora_B.weight": up},
        tmp_path / "adapter_model.safetensors",
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 32, "r": 16}))
    merge_adapter(model, tmp_path)
    assert torch.allclose(model.get_submodule(layer).weight, before + 2.0 * up @ down, atol=1e-6)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> baselines.Workspace:
    return baselines.Workspace.load(tiny_assets("base", tmp_path_factory.mktemp("baselines")), "cpu")


@pytest.mark.parametrize("method", baselines.METHODS)
def test_baseline_builds_a_full_state(workspace, method, monkeypatch):
    options = baselines.settings()
    for name in ("regmean", "regmeanpp", "fisher", "mats", "featcal"):
        options[name]["samples"] = PROMPTS // 2
    options["adamerging"].update(samples=PROMPTS, steps=2)
    options["wudi"]["steps"] = 3
    monkeypatch.setattr(baselines, "settings", lambda: options)
    monkeypatch.setattr(baselines, "CALIBRATION_POOL", PROMPTS)

    value = next(iter(options[method]["grid"].values()))[-2] if options[method].get("grid") else None
    state, details = baselines.run_baseline(method, workspace, seed=0, value=value)
    assert set(state) == set(workspace.pretrained)
    assert all(state[key].shape == workspace.pretrained[key].shape for key in state)
    assert all(bool(torch.isfinite(tensor).all()) for tensor in state.values())
    assert state[EMBEDDING_ALIASES[0]] is state[EMBEDDING]
    assert details.get("value") == value
    assert all(
        parameter.is_leaf for parameter in workspace.model.parameters()
    )  # the scratch model stays usable
    if method == "wa":
        mean = torch.stack([expert[f"{HEAD}.weight"] for expert in workspace.experts]).mean(dim=0)
        assert torch.allclose(state[f"{HEAD}.weight"], mean, atol=1e-6)


def test_command_line_registration():
    parser = argparse.ArgumentParser()
    run.register(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["t5", "--size", "large", "--init", "iso_cts", "--seed", "2", "--freeze-head"])
    assert (args.size, args.init, args.seed, args.freeze_head, args.run) == (
        "large",
        "iso_cts",
        2,
        True,
        run.run,
    )
    args = parser.parse_args(["t5-baseline", "--method", "ta", "--value", "0.3"])
    assert (args.size, args.method, args.value) == ("base", "ta", 0.3)
    assert parser.parse_args(["t5-evaluate", "--checkpoint", "experts"]).run is run.run_evaluate
    assert parser.parse_args(["prepare-t5", "--size", "base", "large"]).size == ["base", "large"]
    for size in ("base", "large"):  # every named initialization has a recorded scale, except weight averaging
        recorded = baselines.selected_values(size)["initializations"]
        assert set(recorded) == set(run.INITIALIZATIONS) - {"wa"}


def test_commands_write_results(assets, tmp_path, monkeypatch):
    """`if-merge t5` and `if-merge t5-baseline` end to end, with the GLUE evaluation stubbed."""
    scores = {"average": 50.0, "per_task": dict.fromkeys(TASKS, 50.0)}
    monkeypatch.setattr(run, "open_assets", lambda args: assets)
    monkeypatch.setattr(run, "evaluate", lambda *args, **kwargs: scores)
    monkeypatch.setattr(run, "evaluate_model", lambda *args, **kwargs: scores)
    parser = argparse.ArgumentParser()
    run.register(parser.add_subparsers(dest="command"))
    common = ["--size", assets.size, "--device", "cpu", "--runs", str(tmp_path)]

    budget = [
        f"calibration.samples={PROMPTS // 2}",
        f"calibration.holdout={PROMPTS // 2}",
        "iteration.rounds=1",
    ]
    overrides = [argument for override in budget for argument in ("--set", override)]
    args = parser.parse_args(["t5", "--init", "wa", "--name", "smoke", *common, *overrides])
    record = args.run(args)
    output = tmp_path / "t5" / assets.size / "if_merge" / "smoke"
    assert json.loads((output / "result.json").read_text())["selected_round"] == record["selected_round"]
    assert [entry["round"] for entry in record["rounds"]] == [0, 1] and record["overrides"] == budget
    assert record["rounds"][1]["holdout_kl"] <= record["rounds"][0]["holdout_kl"]
    assert record["initialization"] == {"name": "wa", "method": "wa", "scale": None}
    merged = load_state(output / "merged.safetensors")
    assert set(merged) == set(model_state(load_model(assets)))

    args = parser.parse_args(
        ["t5", "--init", str(output / "merged.safetensors"), "--no-save", *common, *overrides]
    )
    assert args.run(args)["initialization"]["checkpoint"].endswith("merged.safetensors")

    args = parser.parse_args(["t5-baseline", "--method", "ta", "--value", "0.3", *common])
    record = args.run(args)
    output = tmp_path / "t5" / assets.size / "baselines" / "ta" / "value0.3-seed0"
    assert (output / "merged.safetensors").exists() and (output / "result.json").exists()
    assert (record["hyperparameter"], record["value"], record["selected"]) == ("scale", 0.3, False)
    assert record["validation"] == scores


def test_selection_protocol(workspace, monkeypatch):
    """The best grid value wins, ties go to the earlier one, and only a scale grid is extended."""
    options = baselines.settings()
    options["ta"]["grid"]["scale"] = [0.1, 0.2, 0.3]
    options["regmean"].update(samples=PROMPTS // 2, grid={"off_diagonal_scale": [0.1, 0.5, 0.9]})
    monkeypatch.setattr(baselines, "settings", lambda: options)
    monkeypatch.setattr(baselines, "CALIBRATION_POOL", PROMPTS)
    monkeypatch.setattr(T5Assets, "selection", lambda self, task, seed=0: {})

    def scored(averages):
        averages = iter(averages)
        monkeypatch.setattr(baselines, "evaluate_model", lambda *args, **kwargs: {"average": next(averages)})

    scored([1.0, 2.0, 3.0, 4.0, 3.5, 4.0, 2.0, 99.0])  # 0.4 improves, then three misses in a row
    state, record = baselines.run_baseline("ta", workspace)
    assert (record["value"], record["selected"], record["grid_extended"]) == (0.4, True, True)
    assert record["grid"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    expected = baselines.run_baseline("ta", workspace, value=0.4)[0]
    assert all(torch.equal(state[key], expected[key]) for key in state)

    scored([2.0, 2.0, 1.0])
    assert baselines.run_baseline("ta", workspace)[1]["value"] == 0.1

    scored([1.0, 2.0, 3.0])
    record = baselines.run_baseline("regmean", workspace)[1]
    assert (record["value"], record["grid_extended"]) == (0.9, False)
