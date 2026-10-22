# Batched GGUF projections

`mlx_kquant.projection_pair.KQuantProjectionPair` caches two compatible,
bias-free `KQuantLinear` weights concatenated along the output dimension. Its
call returns two outputs from one existing quantized matmul. It does not decode
or requantize weights and does not fuse the activation. Both projections must
consume the same input and use the same codec and input dimension.

```python
from mlx_kquant.projection_pair import KQuantProjectionPair
pair = KQuantProjectionPair(gate, up, rows=(2048,))
g, u = pair(x, gate, up)
```

The default dispatch is BF16, 2048 flattened input rows. Callers must benchmark
and gate this policy for their own hardware/model. Unsupported rows/dtypes fall
back to the original projections. Floating point equality is shape-dependent;
this API does not promise bitwise equality for arbitrary shapes or codecs.

This is an immutable-inference cache, not a trainable or serializable module.
Keep it outside model parameters. Replacing modules, weights or scales causes
fallback; recreate the pair after any in-place weight edits. The additional
GPU allocation is `pair.nbytes`; original weights remain available for decode.
A caller enabling training must bypass the pair.

On the local M5 Ultra, the Qwen 27B integration enables 24 same-codec MLP pairs
(IQ3_XXS, IQ3_S, IQ4_XS), with 1.686 GiB extra memory. The earlier 128K core
prefill experiment measured 147.43 to 137.37 seconds (7.33% throughput gain).
These measurements exclude loading, tokenization and HTTP overhead. They are
not a general speedup guarantee.
