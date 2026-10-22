"""Inference-only batching of two same-input GGUF linear projections.

This cache is deliberately not an nn.Module: packed copies are runtime state,
not checkpoint parameters. Recreate it after modifying weights in place.
"""

import mlx.core as mx

import mlx_kquant as kq

from .nn import KQuantLinear


class KQuantProjectionPair:
    """Cache concatenated wire bytes without decoding or requantizing weights.

    Callers select measured shapes/dtypes; other inputs use original linears.
    Replaced weights, scales or modules also fall back, avoiding stale caches.
    In-place weight mutation is unsupported. Intended for immutable inference.
    """

    @staticmethod
    def compatible(first, second):
        return (
            type(first) is KQuantLinear
            and type(second) is KQuantLinear
            and first.kquant_type == second.kquant_type
            and first.weight.ndim == second.weight.ndim == 2
            and first.weight.shape[1] == second.weight.shape[1]
            and all(
                "bias" not in p
                and p.weight.dtype == mx.uint8
                and p.scales.dtype == mx.uint8
                and p.scales.size == 1
                for p in (first, second)
            )
        )

    def __init__(self, first, second, *, rows=(2048,), dtype=mx.bfloat16):
        if not self.compatible(first, second):
            raise ValueError(
                "Expected compatible bias-free GGUF KQuantLinear projections"
            )
        self.first, self.second = first, second
        self.weights = (first.weight, second.weight)
        self.scales = (first.scales, second.scales)
        self.codec = first.kquant_type
        self.rows, self.dtype = tuple(rows), dtype
        self.weight = mx.concatenate(self.weights, axis=0)
        self.split = first.weight.shape[0]
        mx.eval(self.weight)

    @property
    def nbytes(self):
        return self.weight.nbytes

    def __call__(self, x, first, second):
        if (
            first is not self.first
            or second is not self.second
            or x.dtype != self.dtype
            or x.size // x.shape[-1] not in self.rows
            or any(
                p.weight is not w
                or p.scales is not s
                or "bias" in p
                or p.kquant_type != self.codec
                for p, w, s in zip(
                    (first, second), self.weights, self.scales, strict=True
                )
            )
        ):
            return first(x), second(x)
        y = kq.quantized_matmul(
            x, self.weight, self.scales[0], self.codec, transpose=True
        )
        return tuple(mx.split(y, [self.split], axis=-1))
