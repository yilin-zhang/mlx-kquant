"""Projection batching preserves outputs and does not reuse replaced weights."""

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType, quants

from mlx_kquant.nn import KQuantLinear
from mlx_kquant.projection_pair import KQuantProjectionPair


def linear(seed, out_dims=64):
    p = KQuantLinear(256, out_dims, False, "q8_0")
    w = np.random.default_rng(seed).normal(size=(out_dims, 256)).astype(np.float32)
    p.weight = mx.array(quants.quantize(w, GGMLQuantizationType.Q8_0))
    return p


def test_pair_and_fallbacks():
    first, second = linear(1), linear(2, 128)
    pair = KQuantProjectionPair(first, second)
    for rows, dtype in [
        (2048, mx.bfloat16),
        (1, mx.bfloat16),
        (65, mx.bfloat16),
        (2048, mx.float32),
    ]:
        x = mx.random.normal((1, rows, 256)).astype(dtype)
        actual = pair(x, first, second)
        for got, expected in zip(actual, (first(x), second(x)), strict=True):
            np.testing.assert_allclose(
                np.array(got.astype(mx.float32)),
                np.array(expected.astype(mx.float32)),
                rtol=0.02,
                atol=0.5,
            )
    # A checkpoint reload or adapter replacing a projection must bypass the copy.
    first.weight = linear(3).weight
    x = mx.random.normal((1, 2048, 256)).astype(mx.bfloat16)
    for a, b in zip(pair(x, first, second), (first(x), second(x)), strict=True):
        np.testing.assert_array_equal(
            np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
        )
    replacement = linear(4)
    a, b = pair(x, replacement, second)
    np.testing.assert_array_equal(
        np.array(a.astype(mx.float32)), np.array(replacement(x).astype(mx.float32))
    )


def test_reject_incompatible():
    first, second = linear(1), linear(2)
    second.bias = mx.zeros((64,))
    with pytest.raises(ValueError):
        KQuantProjectionPair(first, second)
    with pytest.raises(ValueError):
        KQuantProjectionPair(first, KQuantLinear(256, 64, False, "q4_0"))
