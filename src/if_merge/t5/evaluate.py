"""GLUE evaluation of a Flan-T5 model by answer generation (the FusionBench protocol).

Every prompt is padded to 512 tokens and answered by greedy decoding of at most ten
decoder positions. A prediction is correct when the generated text equals the reference
answer exactly; STS-B reports the Spearman correlation (x100) between generated and
reference scores, with unparsable answers counted as 0. The average weights the tasks
equally.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn

from .data import TASKS, T5Assets
from .model import State, load_into, load_model

GENERATION_MAX_LENGTH = 10
PINNED_TRANSFORMERS = "4.51.3"


def check_environment() -> None:
    import transformers

    if transformers.__version__ != PINNED_TRANSFORMERS:
        warnings.warn(
            f"The paper's Flan-T5 scores were measured with Transformers {PINNED_TRANSFORMERS}; "
            f"version {transformers.__version__} may generate different answers.",
            stacklevel=2,
        )


@torch.no_grad()
def generate_answers(
    model: nn.Module, tokenizer, input_ids: torch.Tensor, device: str, batch_size: int = 32
) -> list[str]:
    answers: list[str] = []
    for start in range(0, len(input_ids), batch_size):
        batch = input_ids[start : start + batch_size].to(device).long()
        generated = model.generate(
            batch, attention_mask=(batch != tokenizer.pad_token_id).long(), max_length=GENERATION_MAX_LENGTH
        )
        answers.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return answers


def reference_answers(tokenizer, labels: torch.Tensor) -> list[str]:
    """Decode tokenized reference answers (padding is stored as -100)."""
    special = set(tokenizer.all_special_ids)
    references = []
    for row in labels.tolist():
        tokens = []
        for token in row:
            if token == -100:
                break
            if token > 0 and token not in special:
                tokens.append(token)
        references.append(tokenizer.decode(tokens, skip_special_tokens=True))
    return references


def score(task: str, answers: Sequence[str], references: Sequence[str]) -> float:
    """Exact-match accuracy (%), or 100 x Spearman correlation for STS-B."""
    if task != "stsb":
        return 100.0 * sum(a == r for a, r in zip(answers, references, strict=True)) / max(len(answers), 1)
    from scipy.stats import spearmanr

    def number(text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return 0.0

    correlation = float(spearmanr([number(a) for a in answers], [number(r) for r in references])[0])
    return 0.0 if math.isnan(correlation) else 100.0 * correlation  # nan: constant predictions


def evaluate_model(
    model: nn.Module,
    assets: T5Assets,
    device: str,
    tasks: Sequence[str] = TASKS,
    data: dict[str, dict] | None = None,
    batch_size: int = 32,
) -> dict:
    """Per-task and task-averaged score of ``model``.

    ``data`` maps a task to tokenized ``input_ids`` and ``labels``; the default is the
    GLUE validation split.
    """
    check_environment()
    model = model.to(device).eval()
    per_task = {}
    for task in tasks:
        examples = data[task] if data else assets.validation(task)
        answers = generate_answers(model, assets.tokenizer, examples["input_ids"], device, batch_size)
        per_task[task] = score(task, answers, reference_answers(assets.tokenizer, examples["labels"]))
    return {"average": sum(per_task.values()) / len(per_task), "per_task": per_task}


def evaluate(state: State, assets: T5Assets, device: str, tasks: Sequence[str] = TASKS) -> dict:
    """Validation scores of a merged state dict."""
    model = load_into(load_model(assets), state)
    return evaluate_model(model, assets, device, tasks)
