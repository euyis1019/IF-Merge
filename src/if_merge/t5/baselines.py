"""Merging baselines on Flan-T5 and the protocol that selects their hyperparameters.

Settings and grids are in ``configs/baselines/t5.yaml``. The merging rules shared with
the CLIP benchmark come from ``if_merge.baselines``; what is specific to Flan-T5 is
here:

* Weight-space and spectral rules act on the weight matrices (block matrices and the
  vocabulary head). The other tensors (layer norms, position biases, embedding) take the
  scaled sum of their task vectors for Task Arithmetic and TIES and the expert mean
  otherwise.
* Data-dependent baselines read the same unlabeled prompts as the statistics of
  IF-Merge; the decoder is fed the expert's greedy answer, and statistics are taken over
  valid positions only (see ``if_merge.t5.backend``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.func import functional_call

from ..baselines.fisher import fisher_merge
from ..baselines.fisher import mats as solve_mats
from ..baselines.spectral import iso_c, iso_cts, tsv_m
from ..baselines.spectral import wudi as wudi_matrices
from ..baselines.weight_space import task_arithmetic, ties, weight_average
from ..core.statistics import StatisticsCollector
from .backend import Teacher, build_teacher, forward, valid_rows
from .data import CALIBRATION_POOL, TASKS, T5Assets
from .evaluate import evaluate_model
from .model import (
    EMBEDDING_ALIASES,
    HEAD,
    load_into,
    load_model,
    merged_layers,
    model_state,
    reads_encoder_positions,
    tie_embeddings,
)

CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "baselines"
BATCH_SIZE = 16  # prompts per forward pass when measuring statistics
Tensors = dict[str, torch.Tensor]
Builder = Callable[[float | None], Tensors]  # hyperparameter value -> merged state dict


def settings() -> dict:
    return yaml.safe_load((CONFIGS / "t5.yaml").read_text())


def selected_values(size: str) -> dict:
    """Hyperparameter values of the paper's Flan-T5 results for one model size."""
    return yaml.safe_load((CONFIGS / "t5_selected.yaml").read_text())[size]


@dataclass
class Workspace:
    """The pretrained model, the experts, and a scratch model on the device."""

    assets: T5Assets
    device: str
    model: nn.Module
    pretrained: Tensors
    experts: list[Tensors]  # in task order
    matrix_keys: list[str]  # weights of the merged matrices
    other_keys: list[str]  # every other tensor, the embedding listed once

    @classmethod
    def load(cls, assets: T5Assets, device: str) -> Workspace:
        model = load_model(assets)
        pretrained = model_state(model)
        experts = [model_state(load_model(assets, task)) for task in TASKS]
        matrix_keys = [f"{name}.weight" for name in merged_layers(model)]
        other_keys = [key for key in pretrained if key not in matrix_keys and key not in EMBEDDING_ALIASES]
        return cls(assets, device, model.to(device), pretrained, experts, matrix_keys, other_keys)

    @property
    def layers(self) -> list[str]:
        return [key.removesuffix(".weight") for key in self.matrix_keys]

    def task_vectors(self, keys: Sequence[str]) -> list[Tensors]:
        return [{key: expert[key] - self.pretrained[key] for key in keys} for expert in self.experts]

    def expert_teachers(self, seed: int, samples: int) -> Iterator[Teacher]:
        """For every task in turn: its expert, loaded into the scratch model, answers its prompts."""
        for task, expert in zip(TASKS, self.experts, strict=True):
            load_into(self.model, expert).eval()
            prompts = self.assets.calibration(task, CALIBRATION_POOL, seed)
            prompts = {key: prompts[key][:samples] for key in ("input_ids", "attention_mask")}
            yield build_teacher(self.model, prompts, BATCH_SIZE, self.device)

    def expert_grams(self, seed: int, samples: int) -> list[Tensors]:
        """Input second moments of every merged matrix, measured in each expert on its own task."""
        grams = []
        for teacher in self.expert_teachers(seed, samples):
            masks: Tensors = {}
            with (
                StatisticsCollector(
                    merged_layers(self.model), feedback=False, rows=valid_rows(masks)
                ) as collector,
                torch.no_grad(),
            ):
                for batch in teacher.batches(BATCH_SIZE, self.device):
                    masks["encoder"], masks["decoder"] = batch.attention_mask.bool(), batch.valid.bool()
                    forward(self.model, batch)
            grams.append(collector.finish().A)
        return grams


# ------------------------------------------------- weight-space and spectral
def wudi(task_vectors: Sequence[Tensors], steps: int, learning_rate: float, device: str) -> Tensors:
    """WUDI, with the loss of matrices with many rows (the vocabulary head) in Gram form.

    sum_i ||(M - D_i) D_i^T||^2 / ||D_i||^2 equals sum_i <(M - D_i)^T (M - D_i), D_i^T D_i> / ||D_i||^2,
    which needs [columns, columns] instead of [rows, rows] memory per task.
    """
    tall = [name for name, tensor in task_vectors[0].items() if tensor.shape[0] > 8192]
    rest = [{name: tensor for name, tensor in vector.items() if name not in tall} for vector in task_vectors]
    output = wudi_matrices(rest, steps=steps, learning_rate=learning_rate, device=device)
    for name in tall:
        stack = torch.stack([vector[name].to(device).float() for vector in task_vectors])
        merged = torch.nn.Parameter(stack.sum(dim=0).clone())
        optimizer = torch.optim.Adam([merged], lr=learning_rate, weight_decay=0.0)
        squared_norms = torch.square(torch.norm(stack.reshape(stack.shape[0], -1), p=2, dim=-1))
        grams = stack.transpose(1, 2) @ stack
        with torch.enable_grad():
            for _ in range(steps):
                difference = merged.unsqueeze(0) - stack
                inner = difference.transpose(1, 2) @ difference
                loss = torch.sum((inner * grams).sum(dim=(1, 2)) / squared_norms)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        output[name] = merged.detach().to(task_vectors[0][name].dtype).cpu()
        del stack, grams, merged, optimizer
    return {name: output[name] for name in task_vectors[0]}


def scaled_task_vector(rule: Callable, sum_other: bool = False, on_device: bool = True):
    """A builder for ``pretrained + scale * rule(task vectors)`` on the weight matrices."""

    def make(workspace: Workspace, options: dict, _seed: int) -> Builder:
        arguments = {key: value for key, value in options.items() if key != "grid"}
        if on_device:
            arguments["device"] = workspace.device
        delta = rule(workspace.task_vectors(workspace.matrix_keys), **arguments)
        other = workspace.task_vectors(workspace.other_keys)
        other = task_arithmetic(other) if sum_other else weight_average(other)

        def build(scale: float | None) -> Tensors:
            scale = 1.0 if scale is None else scale
            state = dict(workspace.pretrained)
            for vector, factor in ((delta, scale), (other, scale if sum_other else 1.0)):
                for key, value in vector.items():
                    state[key] = state[key] + factor * value.float()
            return tie_embeddings(state)

        return build

    return make


# ------------------------------------------------------- RegMean and MaTS
def regmean(workspace: Workspace, options: dict, seed: int) -> Builder:
    """RegMean on every merged matrix; all other tensors are the expert mean."""
    grams = workspace.expert_grams(seed, options["samples"])
    device = workspace.device
    mean = {
        key: torch.stack([expert[key] for expert in workspace.experts]).mean(dim=0)
        for key in workspace.pretrained
    }

    @torch.no_grad()
    def build(off_diagonal_scale: float | None) -> Tensors:
        state = dict(mean)
        for layer, key in zip(workspace.layers, workspace.matrix_keys, strict=True):
            numerator = denominator = None
            for expert, task_grams in zip(workspace.experts, grams, strict=True):
                gram = task_grams[layer].to(device).float()
                gram = gram * off_diagonal_scale + torch.diag(gram.diagonal()) * (1.0 - off_diagonal_scale)
                product = gram @ expert[key].to(device).float().t()
                numerator = product if numerator is None else numerator + product
                denominator = gram if denominator is None else denominator + gram
            state[key] = torch.linalg.solve(denominator, numerator).t().contiguous().cpu()
        return tie_embeddings(state)

    return build


def mats(workspace: Workspace, options: dict, seed: int) -> Builder:
    """MaTS: the RegMean objective by conjugate gradient, started from Task Arithmetic."""
    grams = workspace.expert_grams(seed, options["samples"])

    def build(_value: float | None) -> Tensors:
        scale = None if options["ta_scale"] == "selected" else options["ta_scale"]
        initial = build_initialization("ta", workspace, scale)
        merged = solve_mats(
            workspace.experts, grams, workspace.layers, initial, options["iterations"], workspace.device
        )
        return tie_embeddings(merged)

    return build


# ------------------------------------------------------------------ Fisher
def diagonal_fisher(model: nn.Module, teacher: Teacher, device: str) -> Tensors:
    """Diagonal Fisher of the predictions at the valid answer positions, squared per prompt.

    Each prompt back-propagates sum_c sqrt(p_c) log p_c (sqrt(p_c) held constant), summed
    over its valid positions.
    """
    parameters = dict(model.named_parameters())
    for parameter in parameters.values():
        parameter.requires_grad_(True)
    fisher = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
    for prompt in teacher.batches(1, device):
        length = int(prompt.attention_mask.sum())
        prompt.input_ids, prompt.attention_mask = (
            prompt.input_ids[:, :length],
            prompt.attention_mask[:, :length],
        )
        log_probabilities = forward(model, prompt)[prompt.valid.bool()].float().log_softmax(dim=-1)
        loss = (log_probabilities.exp().detach().sqrt() * log_probabilities).sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in parameters.items():
            if parameter.grad is not None:
                fisher[name] += parameter.grad.detach() ** 2
    model.zero_grad(set_to_none=True)
    return {name: (value / len(teacher.input_ids)).cpu() for name, value in fisher.items()}


def fisher(workspace: Workspace, options: dict, seed: int) -> Builder:
    """Fisher Merging with each expert's diagonal Fisher on its own prompts."""

    def build(_value: float | None) -> Tensors:
        fishers = [
            diagonal_fisher(workspace.model, teacher, workspace.device)
            for teacher in workspace.expert_teachers(seed, options["samples"])
        ]
        merged = fisher_merge(workspace.experts, fishers, options["floor"])
        return tie_embeddings({key: merged[key].float() for key in workspace.pretrained})

    return build


# -------------------------------------------------------------- AdaMerging
class ShuffledCycle:
    """Batches of indices from a shuffled pool, reshuffled when exhausted."""

    def __init__(self, size: int, batch_size: int, generator: torch.Generator) -> None:
        self.size, self.batch_size, self.generator = size, batch_size, generator
        self.order, self.position = torch.randperm(size, generator=generator), 0

    def next(self) -> torch.Tensor:
        if self.position + self.batch_size > self.size:
            self.order, self.position = torch.randperm(self.size, generator=self.generator), 0
        index = self.order[self.position : self.position + self.batch_size]
        self.position += self.batch_size
        return index


def adamerging(workspace: Workspace, options: dict, seed: int) -> Builder:
    """Layer-wise AdaMerging with the FusionBench configuration for Flan-T5.

    One unclamped coefficient per expert and parameter tensor, trained to minimize the
    entropy of the batch-averaged prediction at the first decoder position, with dropout
    active. The merged model after the last step is returned.
    """

    def build(_value: float | None) -> Tensors:
        torch.manual_seed(seed)
        model, device = workspace.model, workspace.device
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        names = [name for name, _ in model.named_parameters()]
        pretrained = {name: workspace.pretrained[name].to(device) for name in names}
        deltas = {
            name: torch.stack([expert[name] - workspace.pretrained[name] for expert in workspace.experts]).to(
                device
            )
            for name in names
        }
        coefficients = torch.full(
            (len(TASKS), len(names)), options["initial_coefficient"], device=device, requires_grad=True
        )
        optimizer = torch.optim.Adam([coefficients], lr=options["learning_rate"])
        pools = {task: workspace.assets.calibration(task, options["samples"], seed) for task in TASKS}
        generator = torch.Generator().manual_seed(seed)
        cycles = {task: ShuffledCycle(options["samples"], options["batch_size"], generator) for task in TASKS}

        def merged() -> Tensors:
            return {
                name: pretrained[name]
                + torch.tensordot(coefficients[:, column], deltas[name], dims=([0], [0]))
                for column, name in enumerate(names)
            }

        model.train()
        for _ in range(options["steps"]):
            state = merged()
            for position, task in enumerate(TASKS):
                index = cycles[task].next()
                ids, mask = (
                    pools[task]["input_ids"][index].long(),
                    pools[task]["attention_mask"][index].long(),
                )
                while mask[:, -1].eq(0).all():  # drop the padding shared by the whole batch
                    ids, mask = ids[:, :-1], mask[:, :-1]
                inputs = {
                    "input_ids": ids.to(device),
                    "attention_mask": mask.to(device),
                    "decoder_input_ids": torch.ones(ids.size(0), 1, dtype=torch.long, device=device),
                }
                # the embedding is registered under three names; swapping it once keeps the model intact
                logits = functional_call(model, state, kwargs=inputs, tie_weights=False).logits[:, 0, :]
                probabilities = torch.softmax(logits.mean(dim=0, keepdim=True), dim=-1)
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1).mean()
                entropy.backward(retain_graph=position < len(TASKS) - 1)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            final = {name: value.detach().cpu() for name, value in merged().items()}
        model.eval()
        return tie_embeddings({**workspace.pretrained, **final})

    return build


# --------------------------------------------------------------- RegMean++
class _StopForward(Exception):
    pass


def _without_shared_padding(teacher: Teacher, device: str) -> list[Teacher]:
    """Batches of a teacher's prompts and answers, each trimmed to its longest prompt."""
    batches = []
    for batch in teacher.batches(BATCH_SIZE, "cpu"):
        length = int(batch.attention_mask.sum(dim=1).max())
        batch.input_ids, batch.attention_mask = batch.input_ids[:, :length], batch.attention_mask[:, :length]
        batch.logits = torch.empty(0)
        batches.append(Teacher(*(getattr(batch, name).to(device) for name in batch.__dataclass_fields__)))
    return batches


def _decode(model: nn.Module, batch: Teacher, encoded: torch.Tensor) -> None:
    lengths = batch.valid.sum(dim=1, keepdim=True)
    attention = (torch.arange(batch.valid.shape[1], device=batch.valid.device)[None, :] <= lengths - 1).long()
    model(
        encoder_outputs=(encoded,),
        attention_mask=batch.attention_mask,
        decoder_input_ids=batch.decoder_input_ids,
        decoder_attention_mask=attention,
    )


@contextmanager
def _input_grams(linears: dict[str, nn.Module], prefix: str, masks: Tensors, stop_after: nn.Module | None):
    """Float64 input Grams ``[sum, rows]`` of ``linears``; a forward pass ends after ``stop_after``."""
    moments: dict[str, list] = {name: [0.0, 0] for name in linears}

    def record(name: str):
        def hook(_module, inputs, _output):
            rows = inputs[0][
                masks["encoder" if reads_encoder_positions(prefix + name) else "decoder"]
            ].double()
            moments[name][0] = moments[name][0] + rows.T @ rows
            moments[name][1] += rows.shape[0]

        return hook

    def stop(*_):
        raise _StopForward

    handles = [module.register_forward_hook(record(name)) for name, module in linears.items()]
    if stop_after is not None:
        handles.append(stop_after.register_forward_hook(stop))
    try:
        yield moments
    finally:
        for handle in handles:
            handle.remove()


def _float64_regmean(
    grams: Sequence[torch.Tensor], weights: Sequence[torch.Tensor], off_diagonal_scale: float
):
    left = right = 0.0
    for gram, weight in zip(grams, weights, strict=True):
        gram = off_diagonal_scale * gram + (1.0 - off_diagonal_scale) * torch.diag(torch.diagonal(gram))
        left = left + gram
        right = right + gram @ weight.double().T
    return torch.linalg.solve(left, right).T.float()


def regmeanpp(workspace: Workspace, options: dict, seed: int) -> Builder:
    """RegMean++: merge block by block, feeding each expert block the output of the merged prefix.

    Encoder blocks first, then decoder blocks (cross-attention reads the merged encoder
    output), then the vocabulary head on the merged decoder output. Tensors that are not
    weight matrices are the expert mean.
    """
    experts, device = workspace.experts, workspace.device
    data = [
        _without_shared_padding(teacher, device)
        for teacher in workspace.expert_teachers(seed, options["samples"])
    ]

    @torch.no_grad()
    def build(off_diagonal_scale: float | None) -> Tensors:
        model = load_into(workspace.model, workspace.pretrained).eval()
        for key, value in model.state_dict().items():
            if ".block." not in key:
                value.copy_(torch.stack([expert[key].float() for expert in experts]).mean(dim=0))
        masks: Tensors = {}
        encoded: list[list[torch.Tensor]] = []

        def run(task: int, stack_name: str) -> None:
            for index, batch in enumerate(data[task]):
                masks["encoder"], masks["decoder"] = batch.attention_mask.bool(), batch.valid.bool()
                try:
                    if stack_name == "encoder":
                        model.encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                    else:
                        _decode(model, batch, encoded[task][index])
                except _StopForward:
                    pass

        for stack_name in ("encoder", "decoder"):
            if stack_name == "decoder":  # output of the merged encoder, per task and batch
                encoded = [
                    [
                        model.encoder(
                            input_ids=b.input_ids, attention_mask=b.attention_mask
                        ).last_hidden_state
                        for b in d
                    ]
                    for d in data
                ]
            for block_index, block in enumerate(getattr(model, stack_name).block):
                prefix = f"{stack_name}.block.{block_index}."
                linears = {
                    name: module for name, module in block.named_modules() if isinstance(module, nn.Linear)
                }
                grams: dict[str, list[torch.Tensor]] = {name: [] for name in linears}
                for task, expert in enumerate(experts):
                    block.load_state_dict(
                        {k[len(prefix) :]: v for k, v in expert.items() if k.startswith(prefix)}
                    )
                    with _input_grams(linears, prefix, masks, stop_after=block) as moments:
                        run(task, stack_name)
                    for name in linears:
                        grams[name].append(moments[name][0] / max(moments[name][1], 1))
                merged = {
                    f"{name}.weight": _float64_regmean(
                        grams[name],
                        [e[f"{prefix}{name}.weight"].to(device) for e in experts],
                        off_diagonal_scale,
                    )
                    for name in linears
                }
                for key in block.state_dict():
                    if key not in merged:
                        merged[key] = (
                            torch.stack([e[prefix + key].float() for e in experts]).mean(dim=0).to(device)
                        )
                block.load_state_dict(merged)
        head_grams = []
        for task in range(len(experts)):
            with _input_grams({HEAD: model.lm_head}, "", masks, stop_after=None) as moments:
                run(task, "decoder")
            head_grams.append(moments[HEAD][0] / max(moments[HEAD][1], 1))
        head_weights = [expert[f"{HEAD}.weight"].to(device) for expert in experts]
        model.lm_head.weight.copy_(_float64_regmean(head_grams, head_weights, off_diagonal_scale))
        return model_state(model)

    return build


# ----------------------------------------------------------------- FeatCal
@contextmanager
def _captured_inputs(model: nn.Module, names: Sequence[str], masks: Tensors):
    """Inputs of the named layers at valid positions, for the latest forward pass."""
    modules = dict(model.named_modules())
    rows: Tensors = {}

    def capture(name: str):
        def hook(_module, inputs):
            rows[name] = inputs[0].detach()[masks["encoder" if reads_encoder_positions(name) else "decoder"]]

        return hook

    handles = [modules[name].register_forward_pre_hook(capture(name)) for name in names]
    try:
        yield rows
    finally:
        for handle in handles:
            handle.remove()


def featcal(workspace: Workspace, options: dict, seed: int) -> Builder:
    """FeatCal: calibrate the layer inputs of a merged model toward the experts', in forward order.

    Block by block (then the vocabulary head), every weight matrix is refitted in closed
    form so that the merged model's inputs x_s reproduce the expert's response to
    ``interpolation * x_expert + (1 - interpolation) * x_s``, with a ridge toward an
    extrapolated anchor. Tasks are weighted by the inverse norm of their input moment.
    """
    device, alpha, epsilon = workspace.device, options["interpolation"], 1e-8

    @torch.no_grad()
    def build(_value: float | None) -> Tensors:
        initial = build_initialization("ta", workspace, options["ta_scale"])
        teachers = list(workspace.expert_teachers(seed, options["samples"]))
        experts = [load_model(workspace.assets, task).to(device) for task in TASKS]
        calibrated = load_into(workspace.model, initial).eval()
        layers = merged_layers(calibrated)
        blocks = [f"encoder.block.{index}." for index in range(calibrated.config.num_layers)]
        blocks += [f"decoder.block.{index}." for index in range(calibrated.config.num_decoder_layers)]
        groups = [[name for name in layers if name.startswith(block)] for block in blocks] + [[HEAD]]
        masks: Tensors = {}
        for names in groups:
            # per layer and task: [sum x_s x_s^T, sum x_s x_t^T, rows]
            moments = {name: [[0.0, 0.0, 0] for _ in TASKS] for name in names}
            for task, (teacher, expert) in enumerate(zip(teachers, experts, strict=True)):
                with (
                    _captured_inputs(calibrated, names, masks) as ours,
                    _captured_inputs(expert, names, masks) as theirs,
                ):
                    for batch in teacher.batches(BATCH_SIZE, device):
                        masks["encoder"], masks["decoder"] = batch.attention_mask.bool(), batch.valid.bool()
                        forward(calibrated, batch)
                        forward(expert, batch)
                        for name in names:
                            x_s, x_e = ours[name].float(), theirs[name].float()
                            entry = moments[name][task]
                            entry[0] = entry[0] + x_s.t() @ x_s
                            entry[1] = entry[1] + x_s.t() @ (alpha * x_e + (1.0 - alpha) * x_s)
                            entry[2] += x_s.shape[0]
            for name in names:
                key = f"{name}.weight"
                system = torch.zeros_like(moments[name][0][0])
                right_hand_side = None
                for (second, cross, rows), expert_state in zip(moments[name], workspace.experts, strict=True):
                    norm = (second / rows).norm(p="fro").clamp_min(epsilon)
                    system += (second / rows) / norm
                    term = ((cross / rows) / norm) @ expert_state[key].to(device).t()
                    right_hand_side = term if right_hand_side is None else right_hand_side + term
                anchor = (
                    options["anchor"] * initial[key] + (1.0 - options["anchor"]) * workspace.pretrained[key]
                )
                system += (options["ridge"] + epsilon) * torch.eye(len(system), device=device)
                right_hand_side = right_hand_side + options["ridge"] * anchor.to(device).t()
                layers[name].weight.copy_(torch.linalg.solve(system, right_hand_side).t().contiguous())
        del experts
        return model_state(calibrated)

    return build


# ------------------------------------------------------------------ drivers
BUILDERS: dict[str, Callable[[Workspace, dict, int], Builder]] = {
    "wa": scaled_task_vector(weight_average, on_device=False),
    "ta": scaled_task_vector(task_arithmetic, sum_other=True, on_device=False),
    "ties": scaled_task_vector(ties, sum_other=True, on_device=False),
    "tsv_m": scaled_task_vector(tsv_m),
    "wudi": scaled_task_vector(wudi),
    "iso_c": scaled_task_vector(iso_c),
    "iso_cts": scaled_task_vector(iso_cts),
    "fisher": fisher,
    "regmean": regmean,
    "mats": mats,
    "adamerging": adamerging,
    "regmeanpp": regmeanpp,
    "featcal": featcal,
}
METHODS = tuple(BUILDERS)


def build_initialization(method: str, workspace: Workspace, value: float | None = None) -> Tensors:
    """The merged model of a baseline at ``value`` (default: the value used in the paper)."""
    options = settings()[method] or {}
    if value is None and options.get("grid"):
        value = selected_values(workspace.assets.size)["initializations"][method]
    return BUILDERS[method](workspace, options, 0)(value)


def run_baseline(
    method: str, workspace: Workspace, seed: int = 0, value: float | None = None
) -> tuple[Tensors, dict]:
    """Build one baseline; without ``value``, select its hyperparameter on the labeled split."""
    options = settings()[method] or {}
    build = BUILDERS[method](workspace, options, seed)
    if not options.get("grid"):
        return build(None), {"hyperparameter": None}
    ((name, grid),) = options["grid"].items()
    if value is not None:
        return build(value), {"hyperparameter": name, "value": value, "selected": False}

    data = {task: workspace.assets.selection(task, seed) for task in TASKS}
    sweep: dict[float, dict] = {}
    best: float | None = None
    best_state = None

    def improves(candidate: float) -> bool:
        nonlocal best, best_state
        try:
            state = build(candidate)
        except RuntimeError as error:  # a singular system
            sweep[candidate] = {"error": str(error)[:200]}
            return False
        model = load_into(workspace.model, state)
        sweep[candidate] = evaluate_model(model, workspace.assets, workspace.device, data=data)
        print(f"[{method}] {name} {candidate}: selection score {sweep[candidate]['average']:.2f}", flush=True)
        if best is None or sweep[candidate]["average"] > sweep[best]["average"]:
            best, best_state = candidate, state
            return True
        return False

    grid = list(grid)
    for candidate in grid:
        improves(candidate)
    extension = settings()["selection"]
    extended = name == "scale" and len(grid) > 1 and best == grid[-1]
    if extended:
        spacing, misses, candidate = round(grid[-1] - grid[-2], 4), 0, grid[-1]
        while misses < extension["patience"] and candidate < extension["limit"]:
            candidate = round(candidate + spacing, 4)
            grid.append(candidate)
            misses = 0 if improves(candidate) else misses + 1
    record = {
        "hyperparameter": name,
        "value": best,
        "selected": True,
        "grid": grid,
        "grid_extended": extended,
    }
    return best_state, {**record, "selection_scores": {str(key): scores for key, scores in sweep.items()}}
