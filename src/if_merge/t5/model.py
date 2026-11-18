"""Flan-T5 models and state dicts: which weights are merged, and how they are stored.

The weight matrices IF-Merge fits are the ``nn.Linear`` maps inside the encoder and
decoder blocks (attention q, k, v, o; feed-forward wi_0, wi_1, wo) and the vocabulary
head ``lm_head``. Layer norms, relative position biases and the token embedding keep
the values of the initialization. The embedding is one tensor shared by the encoder
and the decoder; a state dict lists it three times, and ``tie_embeddings`` keeps the
three entries identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from .data import T5Assets

State = Mapping[str, torch.Tensor]

HEAD = "lm_head"
EMBEDDING = "shared.weight"
EMBEDDING_ALIASES = ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight")
BLOCK_LINEAR_SUFFIXES = (".q", ".k", ".v", ".o", ".wi_0", ".wi_1", ".wi", ".wo")


def load_model(assets: T5Assets, task: str | None = None) -> nn.Module:
    """The pretrained model (``task=None``) or the expert of ``task``, in evaluation mode."""
    from transformers import T5ForConditionalGeneration

    return T5ForConditionalGeneration.from_pretrained(assets.model_dir(task)).eval()


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """A float32 CPU copy of the model's state dict, embeddings tied."""
    return tie_embeddings(
        {key: value.detach().float().cpu().clone() for key, value in model.state_dict().items()}
    )


def block_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """The weight matrices inside the encoder and decoder blocks."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and ".block." in name and name.endswith(BLOCK_LINEAR_SUFFIXES)
    }


def merged_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """Every weight matrix that is merged: the block matrices and the vocabulary head."""
    return {**block_layers(model), HEAD: model.lm_head}


def reads_encoder_positions(layer: str) -> bool:
    """Whether the inputs of ``layer`` are encoder positions.

    True for the encoder, and for the key and value projections of the decoder's
    cross-attention, which read the encoder output.
    """
    return layer.startswith("encoder.") or ("EncDecAttention" in layer and layer.endswith((".k", ".v")))


def tie_embeddings(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if EMBEDDING in state:
        for alias in EMBEDDING_ALIASES:
            if alias in state:
                state[alias] = state[EMBEDDING]
    return state


def save_state(state: State, path: str | Path) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # safetensors refuses tensors that share memory, so the embedding aliases are copied
    save_file(
        {
            key: value.contiguous().clone() if key in EMBEDDING_ALIASES else value.contiguous()
            for key, value in state.items()
        },
        str(path),
    )


def load_state(path: str | Path) -> dict[str, torch.Tensor]:
    """A merged model saved by ``save_state``."""
    from safetensors.torch import load_file

    return tie_embeddings({key: value.float() for key, value in load_file(str(path)).items()})


def load_into(model: nn.Module, state: State) -> nn.Module:
    """Load a full state dict; only the embedding aliases may be absent."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if key not in EMBEDDING_ALIASES]
    if missing or unexpected:
        raise ValueError(f"State dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model
