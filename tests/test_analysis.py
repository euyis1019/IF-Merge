"""Pure functions of the analysis scripts."""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("open_clip")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import interpolation_paths  # noqa: E402
import mix_initializations  # noqa: E402
import parameter_distances  # noqa: E402
import spectral_bands  # noqa: E402
from common import interpolate  # noqa: E402


def test_band_slices_follow_the_singular_index():
    slices = spectral_bands.band_slices(768)
    assert (slices["top 1%"].start, slices["top 1%"].stop) == (0, 8)
    assert (slices["1-10%"].start, slices["1-10%"].stop) == (8, 77)
    assert (slices["bottom 50%"].start, slices["bottom 50%"].stop) == (384, 768)
    assert spectral_bands.band_slices(10)["top 1%"] == slice(0, 1)  # a band is never empty


def test_band_coefficient_is_one_when_kept_and_zero_when_removed():
    torch.manual_seed(0)
    initial = torch.randn(6, 6, dtype=torch.float64)
    rows = slice(0, 3)
    assert spectral_bands.coefficient(initial, initial, rows) == pytest.approx(1.0)
    assert spectral_bands.coefficient(initial, -0.5 * initial, rows) == pytest.approx(-0.5)
    assert spectral_bands.coefficient(initial, torch.zeros(6, 6, dtype=torch.float64), rows) == 0.0
    energy = float((initial[rows][:, rows] ** 2).sum())
    assert spectral_bands.coefficient(initial, initial, rows, floor=2 * energy) is None
    permuted = torch.tensor([3, 0, 5])
    assert spectral_bands.coefficient(initial, initial, permuted) == pytest.approx(1.0)


def test_barrier_is_measured_against_the_chord():
    weights = interpolation_paths.WEIGHTS
    line = [80.0 + 5.0 * w for w in weights]
    assert interpolation_paths.barrier(line, True) == pytest.approx(0.0)
    dip = list(line)
    dip[5] -= 1.5
    assert interpolation_paths.barrier(dip, True) == pytest.approx(1.5)
    assert interpolation_paths.barrier(dip, False) == pytest.approx(0.0)  # a loss that dips has no barrier
    bump = [0.5 - 0.1 * w for w in weights]
    bump[3] += 0.2
    assert interpolation_paths.barrier(bump, False) == pytest.approx(0.2)


def test_interpolation_and_block_index():
    a, b = {"w": torch.zeros(2)}, {"w": torch.ones(2)}
    assert interpolate(a, b, 0.3)["w"].tolist() == pytest.approx([0.3, 0.3])
    assert parameter_distances.block_of("transformer.resblocks.11.mlp.c_fc") == 11
    assert parameter_distances.block_of("conv1.linear") is None


def test_weight_matrices_of_an_open_clip_state():
    state = {
        "conv1.weight": torch.randn(8, 3, 4, 4),
        "proj": torch.randn(8, 6),
        "class_embedding": torch.randn(8),
    }
    for block in range(2):
        prefix = f"transformer.resblocks.{block}."
        state[prefix + "attn.in_proj_weight"] = torch.randn(24, 8)
        state[prefix + "attn.in_proj_bias"] = torch.randn(24)
        state[prefix + "attn.out_proj.weight"] = torch.randn(8, 8)
        state[prefix + "mlp.c_fc.weight"] = torch.randn(32, 8)
        state[prefix + "mlp.c_proj.weight"] = torch.randn(8, 32)
    keys = mix_initializations.weight_matrices(state)
    assert (
        len(keys) == 2 * 4 + 2
        and keys[0].endswith("0.attn.in_proj_weight")
        and keys[-2:] == ["conv1.weight", "proj"]
    )
    assert mix_initializations.as_matrix("conv1.weight", state["conv1.weight"]).shape == (8, 48)
    assert mix_initializations.as_matrix("proj", state["proj"]).shape == (6, 8)  # [out, in]
    for key in ("conv1.weight", "proj", keys[0]):
        matrix = mix_initializations.as_matrix(key, state[key])
        assert torch.equal(mix_initializations.from_matrix(key, matrix, state[key]), state[key])


def test_haar_columns_are_orthonormal_and_seeded():
    q = mix_initializations.haar(12, 5, torch.Generator().manual_seed(3))
    assert torch.allclose(q.T @ q, torch.eye(5, dtype=torch.float64), atol=1e-12)
    assert torch.equal(q, mix_initializations.haar(12, 5, torch.Generator().manual_seed(3)))
