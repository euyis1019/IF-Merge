"""Files and data of the Flan-T5 benchmark: eight GLUE tasks as text-to-text problems.

Everything lives under one data root (``--data``, or the ``IF_MERGE_DATA`` environment
variable, default ``./data``), written by ``if-merge prepare-t5``::

    <data>/t5/
      models/<size>/pretrained/           google/flan-t5-<size> (weights and tokenizer)
      models/<size>/<task>/               expert of one task, full weights
      models/large/adapters/<task>/       LoRA adapter the large expert was built from
      glue/<task>/<split>-*.parquet       nyu-mll/glue
      cache/<size>/                       tokenized splits and draws (built on first use)

Prompts, answers and the evaluation protocol follow FusionBench, whose experts these are:
inputs and answers are padded to 512 tokens, MNLI is evaluated on ``validation_matched``
and the STS-B answer is the score printed with one decimal.

Three disjoint parts of the GLUE *training* split are used besides the validation split:

* calibration draw K: the first 512 examples of the permutation ``torch.randperm`` with
  seed K. IF-Merge reads the prompts of the first |C_t| for the statistics and of the
  next |H_t| for step selection; labels are never read.
* selection draw K: 256 labeled examples, used only to select baseline hyperparameters.
  They are the first examples from position 1024 on of the same permutation that are
  not reserved for calibration under any seed (``RESERVED_FOR_CALIBRATION``).
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import torch

from ..paths import default_data_root

SIZES = ("base", "large")
TASKS = ("cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb")
MAX_LENGTH = 512

CALIBRATION_POOL = 512  # examples per calibration draw
SELECTION_EXAMPLES, SELECTION_OFFSET = 256, 1024
#: Training examples our experiments read as calibration prompts, as {seed: length of the
#: prefix of that seed's permutation}. The selection splits stay disjoint from all of them.
RESERVED_FOR_CALIBRATION = {
    "base": {0: 768, 1: 512, 2: 512, 3: 512, 4: 512},
    "large": {0: 512, 1: 512, 2: 512},
}

# Prompts and answers are verbatim from FusionBench, spelling included: the experts were
# fine-tuned on exactly these strings.
PROMPTS = {
    "cola": 'Indicate if the following sentence is grammatically correct or not: "{sentence}". '
    "Answere 'acceptable' or 'unacceptable'.",
    "mnli": "Does the premise: '{premise}' logically imply, contradict, or is neutral to the hypothesis: "
    "'{hypothesis}'? Answere with 'entailment', 'contradiction', or 'neutral'.",
    "mrpc": "Are the following sentences '{sentence1}' and '{sentence2}' conveying the same meaning? "
    "Answere with 'yes' or 'no'.",
    "qnli": "Given the context: '{sentence}', does the question '{question}' have an answer based on the "
    "information provided? Answer with 'yes' or 'no'.",
    "qqp": "Do the questions '{question1}' and '{question2}' have the same intent? "
    "Answere with 'yes' or 'no'.",
    "rte": "Does the text: '{sentence1}' entail that '{sentence2}' is true? Provide 'yes' or 'no'.",
    "sst2": "Given the sentence '{sentence}', determine the sentiment. Is it positive or negative?",
    "stsb": "Consider the sentences '{sentence1}' and '{sentence2}'. On a scale from 1 (completely "
    "different) to 5 (completely similar), rate the similarity.",
}
ANSWERS = {
    "cola": ("unacceptable", "acceptable"),
    "mnli": ("entailment", "neutral", "contradiction"),
    "mrpc": ("no", "yes"),
    "qnli": ("yes", "no"),
    "qqp": ("no", "yes"),
    "rte": ("yes", "no"),
    "sst2": ("negative", "positive"),
}


def validation_split(task: str) -> str:
    return "validation_matched" if task == "mnli" else "validation"


def prompt_and_answer(task: str, example: dict) -> tuple[str, str]:
    """The model input and the reference answer of one GLUE example."""
    prompt = PROMPTS[task].format(**example)
    label = example["label"]
    if task == "stsb":
        return prompt, f"{label:.1f}"
    return prompt, ANSWERS[task][label] if 0 <= label < len(ANSWERS[task]) else ""


def _permutation(examples: int, seed: int) -> torch.Tensor:
    return torch.randperm(examples, generator=torch.Generator().manual_seed(seed))


def calibration_index(examples: int, count: int, seed: int) -> torch.Tensor:
    """Training examples of calibration draw ``seed`` for a split with ``examples`` rows."""
    return _permutation(examples, seed)[:count]


def selection_index(examples: int, size: str, seed: int) -> torch.Tensor:
    """Training examples of selection draw ``seed``, disjoint from everything reserved for calibration."""
    reserved: set[int] = set()
    for calibration_seed, count in RESERVED_FOR_CALIBRATION[size].items():
        reserved.update(calibration_index(examples, count, calibration_seed).tolist())
    candidates = _permutation(examples, seed)[SELECTION_OFFSET:].tolist()
    return torch.tensor([index for index in candidates if index not in reserved][:SELECTION_EXAMPLES])


class T5Assets:
    def __init__(self, size: str, root: str | Path | None = None) -> None:
        if size not in SIZES:
            raise ValueError(f"Unknown size {size!r}; expected one of {SIZES}")
        self.size = size
        self.root = Path(root) if root is not None else default_data_root()
        self.t5 = self.root / "t5"

    # ---------------------------------------------------------------- paths
    def model_dir(self, task: str | None = None) -> Path:
        """Directory of the pretrained model (``task=None``) or of the expert of ``task``."""
        return self.t5 / "models" / self.size / ("pretrained" if task is None else task)

    def adapter_dir(self, task: str) -> Path:
        return self.t5 / "models" / self.size / "adapters" / task

    def glue_dir(self, task: str) -> Path:
        return self.t5 / "glue" / task

    def cache_path(self, task: str, name: str) -> Path:
        return self.t5 / "cache" / self.size / f"{task}__{name}.pt"

    def missing(self, tasks=TASKS) -> list[Path]:
        paths = [self.model_dir() / "config.json", self.model_dir() / "tokenizer.json"]
        paths += [self.model_dir(task) / "config.json" for task in tasks]
        paths += [path for path in map(self.glue_dir, tasks) if not any(path.glob("train-*.parquet"))]
        return [path for path in paths if not path.exists()]

    # -------------------------------------------------------------- loaders
    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.model_dir())

    def validation(self, task: str) -> dict:
        """The GLUE validation split: ``input_ids`` and ``labels`` (tokenized reference answers)."""
        split = validation_split(task)
        return self._cached(
            task, split, lambda: self._encode(task, self._examples(task, split), answers=True)
        )

    def calibration(self, task: str, count: int = CALIBRATION_POOL, seed: int = 0) -> dict:
        """Unlabeled prompts of calibration draw ``seed``: ``input_ids`` and ``attention_mask``."""

        def build() -> dict:
            examples = self._examples(task, "train")
            index = calibration_index(len(examples), count, seed)
            return {**self._encode(task, examples.take(index.tolist()), answers=False), "index": index}

        return self._cached(task, f"calibration_n{count}_seed{seed}", build)

    def selection(self, task: str, seed: int = 0) -> dict:
        """Labeled training examples for selecting baseline hyperparameters."""

        def build() -> dict:
            examples = self._examples(task, "train")
            index = selection_index(len(examples), self.size, seed)
            return {**self._encode(task, examples.take(index.tolist()), answers=True), "index": index}

        return self._cached(task, f"selection_seed{seed}", build)

    # ------------------------------------------------------------- internals
    def _cached(self, task: str, name: str, build) -> dict:
        path = self.cache_path(task, name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(build(), path)
        return torch.load(path, map_location="cpu", weights_only=True)

    def _examples(self, task: str, split: str):
        """One split of a task as a pyarrow table, in the order of the parquet files."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        files = sorted(self.glue_dir(task).glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {split} split in {self.glue_dir(task)}. Run `if-merge prepare-t5`.")
        return pa.concat_tables([pq.read_table(file) for file in files])

    def _encode(self, task: str, examples, answers: bool) -> dict:
        prompts, references = zip(
            *(prompt_and_answer(task, row) for row in examples.to_pylist()), strict=True
        )
        options = {
            "padding": "max_length",
            "truncation": True,
            "max_length": MAX_LENGTH,
            "return_tensors": "pt",
        }
        encoded = self.tokenizer(list(prompts), **options)
        data = {
            "input_ids": encoded["input_ids"].to(torch.int32),
            "attention_mask": encoded["attention_mask"].to(torch.int8),
        }
        if answers:
            labels = self.tokenizer(list(references), **options)["input_ids"]
            labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels.to(torch.int32)
        return data
