# Evidence Policy

The public pull requests are implementation provenance, not the article's
chapter structure and not its final benchmark dataset.

| Implementation thread | Source PR | Used to support |
|---|---|---|
| Online FP8/NF4 framework | [#16](#q-spa-system/pull/16) | Why generic quantization APIs do not guarantee end-to-end acceleration |
| MiniMax-H3 quantized loading | [#25](#q-spa-system/pull/25) | H3 module policy, CPU-first loading, and spawn constraints |
| FP8 Sol-Attn kernel | [#30](#q-spa-system/pull/30) | Shared QKV quantization and the SM90 sparse-attention path |
| Ulysses composition | [#35](#q-spa-system/pull/35) | Post-communication quantization and distributed tensor ownership |
| Lossless SP optimization | [#37](#q-spa-system/pull/37) | Communication/attention overlap and fused Ulysses data movement |
| Attention smoothing | [#40](#q-spa-system/pull/40) | K/V centering, V correction, and fused quality controls |
| Adapter-aware FP8 | [#44](#q-spa-system/pull/44) | Merge-before-quantize ordering and hybrid adapter semantics |

Final claims must point to raw records from the new flagship experiment. The PR
links above may establish that a mechanism exists, but cannot supply the final
speedup or quality conclusion.
