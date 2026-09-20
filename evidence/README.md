# Evidence Policy

The article separates method claims from the benchmark records that support
the final numbers. Each headline result points to a raw JSON record in the
corresponding experiment directory; historical exploratory runs are not used
as headline evidence.

| Evidence group | Directory | Used to support |
|---|---|---|
| Four-GPU end-to-end comparison | `experiments/h100-4gpu-e2e/raw/` | Base H3 latency, throughput, and peak memory |
| Adapter comparison | `experiments/adapter-suite/raw/` | Turbo LoRA and FastH3 results |
| Quality controls | `experiments/quality-suite/raw/` | Tensor error, audio, and video quality metrics |

The article reports the measurement protocol next to each result and does not
promote incomplete or unmatched runs to headline claims.
