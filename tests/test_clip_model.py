"""Exposing the hidden linear maps of the CLIP visual encoder must not change its function."""

import pytest
import torch

open_clip = pytest.importorskip("open_clip")

from if_merge.clip.model import (  # noqa: E402
    build_exposed_visual,
    build_visual,
    linear_layers,
    open_clip_state,
    transformer_blocks,
    visual_state,
)

ARCHITECTURE = "ViT-B-32"


@pytest.fixture(scope="module")
def reference():
    torch.manual_seed(0)
    return build_visual(ARCHITECTURE)  # randomly initialized


def test_exposed_encoder_is_equivalent_and_round_trips(reference):
    state = {f"visual.{name}": value for name, value in reference.state_dict().items()}
    assert visual_state(state).keys() == reference.state_dict().keys()
    exposed = build_exposed_visual(ARCHITECTURE, state)
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(reference(images), exposed(images), atol=1e-4)
    restored = open_clip_state(exposed)
    assert restored.keys() == reference.state_dict().keys()
    assert all(torch.equal(restored[name], value) for name, value in reference.state_dict().items())


def test_every_weight_matrix_is_a_linear_layer(reference):
    layers = linear_layers(build_exposed_visual(ARCHITECTURE, reference.state_dict()))
    blocks = transformer_blocks(layers)
    assert len(layers) == 12 * 6 + 2  # q, k, v, out, two MLP maps per block; patch embedding; projection
    assert list(blocks) == [-1, *range(12)]
    assert sorted(blocks[-1]) == ["conv1.linear", "proj_linear"]
    assert not linear_layers(
        build_exposed_visual(ARCHITECTURE, reference.state_dict(), boundary=False)
    ).keys() & {"conv1.linear", "proj_linear"}
