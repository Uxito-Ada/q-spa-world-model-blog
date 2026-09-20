# Q-SPA: Quantized Sparse-Parallel Attention for World Models

**Co-designing FP8 quantization, dynamic block sparsity, and distributed attention.**

Q-SPA is a system design for quantized, sparse, and parallel attention in
compute-intensive world-model diffusion transformers. This article uses
MiniMax-H3 to show how it combines FP8 Linear, FP8 Sol-Attn, attention
smoothing, and distributed execution across Base H3 and adapter variants.

- [Read the English article](BLOG.md)
- [阅读中文版](BLOG.zh.md)
- [Open the English HTML edition](site/index.html)
- [打开中文 HTML 版](site/index.zh.html)
- [Inspect and reproduce the benchmark](experiments/h100-4gpu-e2e/README.md)
- [Edit an individual section](AUTHORING.md)

## Article map

| Section | Focus |
|---|---|
| [Introduction](sections/00-introduction/README.md) | Q-SPA and the MiniMax-H3 optimization result |
| [Why co-design](sections/01-why-co-design/README.md) | Why FP8 and sparse attention need one design |
| [Q-SPA implementation](sections/02-system-overview/README.md) | Layout-aware FP8 Sol-Attn and distributed execution |
| [Quality-aware FP8](sections/03-quality-and-scale/README.md) | Attention-specific quality control |
| [Results](sections/04-evaluation/README.md) | External performance baseline and generated output |
| [Conclusion](sections/05-lessons/README.md) | The unified Q-SPA inference path |

Each directory owns publishable prose, section metadata, and its media. The
English source is `README.md`; the matching Chinese source is `README.zh.md`.
Both assembled articles are generated with:

    python scripts/build_blog.py
    python scripts/validate_repo.py

The published article is generated from the section directories so each part
can be reviewed and refined without editing one monolithic document.
