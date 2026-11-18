"""IF-Merge backend for Flan-T5 on text-to-text tasks.

A prompt has one prediction distribution per position of its answer. The expert of each
task is the teacher: it answers every unlabeled prompt by greedy decoding, and its
answer is then fed back as the decoder input (teacher forcing), so that the expert and
the merged model predict every answer position from the same prefix. Positions up to
and including the end-of-sequence token are the valid prediction positions V(x).

Statistics are accumulated over valid positions only: non-padding positions for layers
that read the encoder sequence (including the key and value projections of
cross-attention) and valid prediction positions for layers that read the decoder
sequence. The feedback loss is the teacher-student KL of every valid prediction
position divided by that position's chi-square scale. The vocabulary head has one
output per vocabulary entry, so it is fitted with G = I, or kept frozen.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import IFMergeConfig
from ..core.feedback import feedback_loss
from ..core.statistics import StatisticsCollector, TaskStatistics, train_only
from .data import TASKS, T5Assets
from .model import (
    HEAD,
    State,
    block_layers,
    load_into,
    load_model,
    merged_layers,
    model_state,
    reads_encoder_positions,
)

ANSWER_MAX_LENGTH = 10  # decoder positions generated per answer, start token included
G_ESTIMATORS = ("normalized_kl", "kl", "mc_fisher", "identity")


@dataclass
class Teacher:
    """Unlabeled prompts of one task with the expert's answers and predictions. CPU tensors."""

    input_ids: torch.Tensor  # [N, L] prompt tokens
    attention_mask: torch.Tensor  # [N, L] 1 on prompt tokens, 0 on padding
    decoder_input_ids: torch.Tensor  # [N, T] the expert's greedy answer, shifted right
    valid: torch.Tensor  # [N, T] 1 on prediction positions up to and including end-of-sequence
    logits: torch.Tensor  # [N, T, vocabulary] float16, the expert's logits under teacher forcing

    def batches(self, batch_size: int, device: str) -> Iterator[Teacher]:
        for start in range(0, len(self.input_ids), batch_size):
            batch = slice(start, start + batch_size)
            yield Teacher(*(getattr(self, name)[batch].to(device) for name in self.__dataclass_fields__))


def forward(model: nn.Module, batch: Teacher) -> torch.Tensor:
    """Logits [B, T, vocabulary] of ``model`` on the expert's answer prefix."""
    answer_lengths = batch.valid.sum(dim=1, keepdim=True)
    positions = torch.arange(batch.valid.shape[1], device=batch.valid.device)[None, :]
    return model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        decoder_input_ids=batch.decoder_input_ids,
        decoder_attention_mask=(positions <= answer_lengths - 1).long(),
    ).logits


@torch.no_grad()
def build_teacher(expert: nn.Module, prompts: dict, batch_size: int, device: str) -> Teacher:
    """Let the expert answer ``prompts`` greedily, then record its logits under teacher forcing."""
    expert.eval()
    pad, end = expert.config.pad_token_id, expert.config.eos_token_id
    chunks: list[Teacher] = []
    for start in range(0, len(prompts["input_ids"]), batch_size):
        mask = prompts["attention_mask"][start : start + batch_size]
        length = int(mask.sum(dim=1).max())  # drop the padding shared by the whole batch
        ids = prompts["input_ids"][start : start + batch_size, :length].long().to(device)
        mask = mask[:, :length].long().to(device)
        generated = expert.generate(ids, attention_mask=mask, max_length=ANSWER_MAX_LENGTH)
        if generated.shape[1] < 2:
            generated = torch.cat([generated, torch.full_like(generated[:, :1], end)], dim=1)
        decoder_input, target = generated[:, :-1], generated[:, 1:]
        is_end = (target == end).long()
        valid = ((torch.cumsum(is_end, dim=1) - is_end) == 0).long()
        batch = Teacher(ids, mask, decoder_input, valid, torch.empty(0))
        batch.logits = forward(expert, batch).to(torch.float16)
        chunks.append(Teacher(*(getattr(batch, name).cpu() for name in batch.__dataclass_fields__)))

    def join(name: str, fill: int) -> torch.Tensor:
        tensors = [getattr(chunk, name) for chunk in chunks]
        length = max(tensor.shape[1] for tensor in tensors)
        tail = (0, 0) if name == "logits" else ()
        return torch.cat([F.pad(t, (*tail, 0, length - t.shape[1]), value=fill) for t in tensors])

    return Teacher(
        join("input_ids", pad), join("attention_mask", 0), join("decoder_input_ids", pad),
        join("valid", 0), join("logits", 0),
    )  # fmt: skip


def valid_rows(masks: dict[str, torch.Tensor]):
    """Row selector for ``StatisticsCollector``: the valid positions of the current batch.

    ``masks`` is shared with the caller, which sets ``masks["encoder"]`` and ``masks["decoder"]``
    (boolean [batch, positions]) before every forward pass.
    """

    def rows(layer: str, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[masks["encoder" if reads_encoder_positions(layer) else "decoder"]]

    return rows


class T5Backend:
    """Owns the merged model, the experts' weights and predictions, and the calibration prompts."""

    incremental_step = True  # the step is applied as W + step (X - W)

    def __init__(
        self,
        assets: T5Assets,
        initial_state: State,
        config: IFMergeConfig,
        device: str = "cuda",
        batch_size: int = 16,
        freeze_head: bool = False,
        tasks: Sequence[str] = TASKS,
    ) -> None:
        if config.g.estimator not in G_ESTIMATORS:
            raise ValueError(
                f"g.estimator must be one of {list(G_ESTIMATORS)} for T5; got {config.g.estimator!r}"
            )
        if config.selection.scope != "global":
            raise ValueError("T5 selects one step size for the full model: selection.scope must be 'global'.")
        self.tasks = tuple(tasks)
        self.device = device
        self.batch_size = batch_size
        self.seed = config.calibration.seed
        samples, total = config.calibration.samples, config.calibration.samples + config.calibration.holdout

        self.model = load_model(assets)
        self.layers = block_layers(self.model) if freeze_head else merged_layers(self.model)
        self.blocks = {"model": list(self.layers)}  # one step size for the full model
        self.activation_only = frozenset({HEAD} & set(self.layers))

        # Unlabeled prompts: the first `samples` are C_t, the next `holdout` are H_t.
        self.calibration: dict[str, Teacher] = {}
        self.holdout: dict[str, Teacher] = {}
        self._expert_weights: dict[str, list[torch.Tensor]] = {name: [] for name in self.layers}
        for task in self.tasks:
            prompts = assets.calibration(task, total, self.seed)
            if len(prompts["input_ids"]) < total:
                raise ValueError(
                    f"{task}: {len(prompts['input_ids'])} calibration prompts, {total} requested"
                )
            expert = load_model(assets, task).to(device)
            for part, rows in ((self.calibration, slice(0, samples)), (self.holdout, slice(samples, total))):
                selected = {key: prompts[key][rows] for key in ("input_ids", "attention_mask")}
                part[task] = build_teacher(expert, selected, batch_size, device)
            expert_layers = merged_layers(expert)
            for name in self.layers:
                self._expert_weights[name].append(expert_layers[name].weight.detach().float().cpu().clone())
            del expert, expert_layers
        load_into(self.model, initial_state).to(device).eval()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ protocol
    def expert_weights(self, layer: str) -> list[torch.Tensor]:
        return self._expert_weights[layer]

    def state(self) -> dict[str, torch.Tensor]:
        return model_state(self.model)

    @torch.no_grad()
    def holdout_kl(self) -> float:
        """KL averaged over the valid prediction positions of each task, then over tasks."""
        per_task = []
        for task in self.tasks:
            total, positions = 0.0, 0
            for batch in self.holdout[task].batches(self.batch_size, self.device):
                valid = batch.valid.bool()
                log_student = forward(self.model, batch)[valid].float().log_softmax(dim=-1)
                log_teacher = batch.logits[valid].float().log_softmax(dim=-1)
                total += float((log_teacher.exp() * (log_teacher - log_student)).sum())
                positions += int(valid.sum())
            per_task.append(total / max(positions, 1))
        return sum(per_task) / len(per_task)

    def collect_statistics(self, config: IFMergeConfig, round_index: int) -> list[TaskStatistics]:
        estimator = config.g.estimator
        feedback = estimator != "identity"
        fitted_with_g = {
            name: layer for name, layer in self.layers.items() if name not in self.activation_only
        }
        if feedback:
            train_only(self.model, fitted_with_g.values())
        statistics = []
        for task in self.tasks:
            generator = torch.Generator(device=self.device).manual_seed(self.seed + round_index)
            masks: dict[str, torch.Tensor] = {}
            collector = StatisticsCollector(
                self.layers,
                feedback,
                rows=valid_rows(masks),
                inputs_only=self.activation_only,  # no vocabulary-sized output moment for the head
            )
            with collector, torch.set_grad_enabled(feedback):
                for batch in self.calibration[task].batches(self.batch_size, self.device):
                    masks["encoder"], masks["decoder"] = batch.attention_mask.bool(), batch.valid.bool()
                    logits = forward(self.model, batch)[masks["decoder"]]  # [valid positions, vocabulary]
                    teacher_logits = batch.logits[masks["decoder"]].float()
                    if estimator in ("normalized_kl", "kl"):
                        feedback_loss(teacher_logits, logits, estimator == "normalized_kl").sum().backward()
                    elif estimator == "mc_fisher":
                        probabilities = logits.detach().softmax(dim=-1)
                        for sample in range(config.g.mc_samples):
                            labels = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
                            F.cross_entropy(logits, labels, reduction="sum").backward(
                                retain_graph=sample < config.g.mc_samples - 1
                            )
                    for parameter in self.model.parameters():
                        parameter.grad = None
            statistics.append(collector.finish())
            del collector
        train_only(self.model, ())
        return statistics
