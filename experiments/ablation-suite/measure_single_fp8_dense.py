#!/usr/bin/env python3
"""Run the validated single-GPU FP8 dense MiniMax-H3 profile and sample GPU memory."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path("/data/heyang/TeleFuser-minimax-h3-quant-latest")
PYTHON = Path("/data/zuoxin/workspace/TeleFuser/.venv/bin/python")
OUTPUT = Path("/tmp/qspa-base-h3-fp8-dense.mp4")
REPORT = ROOT / "experiments/ablation-suite/raw/qspa-fp8-dense-1gpu.json"


def sample_memory(stop: threading.Event, samples: list[int]) -> None:
    while not stop.is_set():
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            samples.append(int(result.stdout.strip().splitlines()[0]))
        except (IndexError, ValueError):
            pass
        stop.wait(0.1)


def main() -> None:
    command = [
        str(PYTHON),
        str(RUNTIME / "tools/validation/benchmark_minimax_h3_quantization.py"),
        "--backend",
        "tf-kernel-fp8",
        "--model-root",
        "/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3",
        "--prompt",
        "Steam rises from the ramen while the family talks in the background.",
        "--duration",
        "5",
        "--steps",
        "50",
        "--seed",
        "0",
        "--aspect-ratio",
        "16:9",
        "--device",
        "cuda:0",
        "--output",
        str(OUTPUT),
        "--metrics-json",
        "/tmp/qspa-base-h3-fp8-dense.metrics.json",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = str(RUNTIME) + os.pathsep + env.get("PYTHONPATH", "")
    samples: list[int] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(stop, samples), daemon=True)
    started = time.perf_counter()
    sampler.start()
    try:
        completed = subprocess.run(command, cwd=RUNTIME, env=env, text=True, check=True)
    finally:
        stop.set()
        sampler.join(timeout=2)
    elapsed = time.perf_counter() - started
    source = json.loads(Path("/tmp/qspa-base-h3-fp8-dense.metrics.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "profile": "single_gpu_fp8_dense",
        "backend": "tf-kernel-fp8",
        "attention": "FlashAttention 4 dense",
        "gpu_count": 1,
        "generation_seconds": source["generation_seconds"],
        "denoising_seconds": source["runtime_metrics"].get("denoising_seconds"),
        "peak_memory_mib": max(samples) if samples else None,
        "wall_seconds_including_runner": elapsed,
        "source_metrics": source,
        "command": command,
        "runtime_root": str(RUNTIME),
        "model_root": source["model_root"],
        "output": str(OUTPUT),
        "returncode": completed.returncode,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
