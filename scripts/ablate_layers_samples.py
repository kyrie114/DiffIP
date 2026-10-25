"""Ablation: how do the number of layers and samples affect the margin?

Margin = similarity(positive real-SFT pair) - similarity(negative cross-family
pair), using SmolLM2-135M vs SmolLM2-135M-Instruct (positive) and
SmolLM2-135M vs Qwen2.5-0.5B (negative). Baseline config (L=4, N=8) gives
margin ~0.18.

Configurations (all at perturbation scale 2e-3, seed 123):
  L sweep at N=8 : L=8, L=16 (evenly spaced layers of the 30-layer model)
  N sweep at L=4 : N=4, N=16
  combined       : L=8, N=16

Usage:
    HF_ENDPOINT=https://hf-mirror.com python -m scripts.ablate_layers_samples
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffip import DiffIPMetric
from diffip.llm import LLMRepresentationCollector

PROMPT = "The quick brown fox jumps over the lazy dog."
SCALE = 2e-3
SEED = 123
N_COMPONENTS = 20

VICTIM_ID = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
QWEN_ID = "Qwen/Qwen2.5-0.5B"
VICTIM_TOTAL_LAYERS = 30
QWEN_TOTAL_LAYERS = 24

CONFIGS = [
    (4, 8),   # baseline (reference values already measured)
    (8, 8),
    (16, 8),
    (4, 4),
    (4, 16),
    (8, 16),
]


def load(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    return model, tokenizer


def even_layers(total: int, count: int) -> list[int]:
    indices = np.linspace(0, total - 1, count)
    return sorted({int(round(i)) for i in indices})


def fingerprint(model_id: str, layers, n_samples: int):
    model, tokenizer = load(model_id)
    collector = LLMRepresentationCollector(
        model,
        tokenizer,
        n_samples=n_samples,
        perturbation_scale=SCALE,
        layers=layers,
        seed=SEED,
    )
    representations = collector.collect(PROMPT).representations
    del model
    return representations


def main() -> None:
    metric = DiffIPMetric(n_components=N_COMPONENTS)
    print(
        f"{'config':>10} | {'positive':>8} | {'negative':>8} | {'margin':>7}",
        flush=True,
    )
    print("-" * 45, flush=True)

    for n_layers, n_samples in CONFIGS:
        victim_layers = even_layers(VICTIM_TOTAL_LAYERS, n_layers)
        qwen_layers = even_layers(QWEN_TOTAL_LAYERS, n_layers)

        victim = fingerprint(VICTIM_ID, victim_layers, n_samples)
        instruct = fingerprint(INSTRUCT_ID, victim_layers, n_samples)
        qwen = fingerprint(QWEN_ID, qwen_layers, n_samples)

        r_pos = metric.compare(instruct, victim)
        r_neg = metric.compare(qwen, victim)
        margin = r_pos.similarity - r_neg.similarity
        print(
            f"({n_layers:>2},{n_samples:>2}) | {r_pos.similarity:>8.4f} | "
            f"{r_neg.similarity:>8.4f} | {margin:>7.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
