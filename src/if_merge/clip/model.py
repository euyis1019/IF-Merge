"""CLIP visual encoder with every weight matrix exposed as an ``nn.Linear``.

IF-Merge fits weight matrices and reads their inputs and output gradients through
module hooks. In open_clip, three kinds of linear maps are not ``nn.Linear`` modules:
the packed query/key/value projection of ``nn.MultiheadAttention``, the patch
embedding (a strided convolution) and the output projection (a bare parameter).
``expose_linear_maps`` rewrites them as ``nn.Linear`` without changing the function;
``open_clip_state`` converts the weights back to the original checkpoint layout.
"""

from __future__ import annotations

import re
import types
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

State = Mapping[str, torch.Tensor]


def visual_state(state: State) -> dict[str, torch.Tensor]:
    """The visual tower's entries of a CLIP state dict, without the ``visual.`` prefix."""
    state = {key.removeprefix("model."): value for key, value in state.items()}
    if any(key.startswith("visual.") for key in state):
        return {k.removeprefix("visual."): v for k, v in state.items() if k.startswith("visual.")}
    return state


def build_visual(architecture: str, state: State | None = None) -> nn.Module:
    """The open_clip visual tower, optionally loaded from a CLIP or visual state dict.

    The experts were fine-tuned from OpenAI CLIP, which uses QuickGELU. The activation
    leaves no trace in a state dict, so it is fixed here.
    """
    import open_clip

    model = open_clip.create_model(architecture, pretrained=None, force_quick_gelu=True)
    visual = model.visual
    if state is not None:
        visual.load_state_dict(visual_state(state), strict=True)
    return visual.eval()


class SplitAttention(nn.Module):
    """``nn.MultiheadAttention`` (self-attention) with separate q/k/v/out projections."""

    def __init__(self, attention: nn.MultiheadAttention) -> None:
        super().__init__()
        dim = attention.embed_dim
        self.embed_dim, self.num_heads, self.head_dim = dim, attention.num_heads, dim // attention.num_heads
        self.batch_first = bool(getattr(attention, "batch_first", False))
        bias = attention.in_proj_bias is not None
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=attention.out_proj.bias is not None)
        with torch.no_grad():
            for index, projection in enumerate((self.q_proj, self.k_proj, self.v_proj)):
                rows = slice(index * dim, (index + 1) * dim)
                projection.weight.copy_(attention.in_proj_weight[rows])
                if bias:
                    projection.bias.copy_(attention.in_proj_bias[rows])
            self.out_proj.load_state_dict(attention.out_proj.state_dict())

    def forward(self, query, key, value, need_weights=False, attn_mask=None, **_):
        if self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        length, batch, _ = query.shape

        def heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.reshape(-1, batch * self.num_heads, self.head_dim).transpose(0, 1)

        output = F.scaled_dot_product_attention(
            heads(self.q_proj(query)), heads(self.k_proj(key)), heads(self.v_proj(value)), attn_mask=attn_mask
        )
        output = self.out_proj(output.transpose(0, 1).reshape(length, batch, self.embed_dim))
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, None


class PatchLinear(nn.Module):
    """A convolution whose stride equals its kernel size, as unfold + ``nn.Linear``."""

    def __init__(self, convolution: nn.Conv2d) -> None:
        super().__init__()
        self.patch = convolution.kernel_size[0]
        self.linear = nn.Linear(
            convolution.in_channels * self.patch**2,
            convolution.out_channels,
            bias=convolution.bias is not None,
        )
        with torch.no_grad():
            self.linear.weight.copy_(convolution.weight.reshape(convolution.out_channels, -1))
            if convolution.bias is not None:
                self.linear.bias.copy_(convolution.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        patches = F.unfold(images, kernel_size=self.patch, stride=self.patch)
        embedded = self.linear(patches.transpose(1, 2))
        return embedded.transpose(1, 2).reshape(batch, -1, height // self.patch, width // self.patch)


def _forward_with_linear_projection(self, images):
    pooled, tokens = self._pool(self.transformer(self._embeds(images)))
    pooled = self.proj_linear(pooled)
    return (pooled, tokens) if getattr(self, "output_tokens", False) else pooled


def expose_linear_maps(visual: nn.Module, boundary: bool = True) -> nn.Module:
    """Rewrite, in place, the hidden linear maps of ``visual`` as ``nn.Linear`` modules.

    ``boundary`` also rewrites the patch embedding and the output projection.
    """
    for parent in list(visual.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                setattr(parent, name, SplitAttention(child))
    unsplit = [
        name
        for name, module in visual.named_modules()
        if name.endswith(".attn") and not isinstance(module, SplitAttention)
    ]
    if unsplit:
        raise RuntimeError(
            f"{len(unsplit)} attention modules are not nn.MultiheadAttention and were not split "
            "(e.g. " + unsplit[0] + "); this code expects open-clip-torch>=3.3,<4."
        )
    if boundary:
        if isinstance(visual.conv1, nn.Conv2d):
            visual.conv1 = PatchLinear(visual.conv1)
        if visual.proj is not None:
            projection = visual.proj  # [width, output]
            linear = nn.Linear(projection.shape[0], projection.shape[1], bias=False)
            with torch.no_grad():
                linear.weight.copy_(projection.transpose(0, 1))
            visual.proj_linear = linear.to(projection.device, projection.dtype)
            visual.proj = None
            visual.forward = types.MethodType(_forward_with_linear_projection, visual)
    return visual


def build_exposed_visual(architecture: str, state: State | None = None, boundary: bool = True) -> nn.Module:
    return expose_linear_maps(build_visual(architecture, state), boundary)


def linear_layers(visual: nn.Module) -> dict[str, nn.Linear]:
    """The weight matrices IF-Merge fits, by module name."""
    return {name: module for name, module in visual.named_modules() if isinstance(module, nn.Linear)}


def transformer_blocks(layer_names) -> dict[int, list[str]]:
    """Layers grouped by Transformer block; the boundary maps form block -1."""
    blocks: dict[int, list[str]] = {}
    for name in layer_names:
        match = re.search(r"resblocks\.(\d+)\.", name)
        blocks.setdefault(int(match.group(1)) if match else -1, []).append(name)
    return dict(sorted(blocks.items()))


def open_clip_state(visual: nn.Module | State) -> dict[str, torch.Tensor]:
    """State dict of an exposed encoder in the original open_clip visual layout (on the CPU)."""
    exposed = visual.state_dict() if isinstance(visual, nn.Module) else visual
    exposed = {name: tensor.detach().float().cpu() for name, tensor in exposed.items()}
    state: dict[str, torch.Tensor] = {}
    for name, tensor in exposed.items():
        if name.endswith(
            (".attn.k_proj.weight", ".attn.v_proj.weight", ".attn.k_proj.bias", ".attn.v_proj.bias")
        ):
            continue
        if ".attn.q_proj." in name:
            prefix, kind = name.split("q_proj.")
            parts = [exposed[f"{prefix}{p}_proj.{kind}"] for p in "qkv"]
            state[f"{prefix}in_proj_{kind}"] = torch.cat(parts)
        elif name == "conv1.linear.weight":
            patch = int(round((tensor.shape[1] // 3) ** 0.5))
            state["conv1.weight"] = tensor.reshape(tensor.shape[0], 3, patch, patch).clone()
        elif name == "conv1.linear.bias":
            state["conv1.bias"] = tensor.clone()
        elif name == "proj_linear.weight":
            state["proj"] = tensor.transpose(0, 1).contiguous()
        else:
            state[name] = tensor.clone()
    return state
