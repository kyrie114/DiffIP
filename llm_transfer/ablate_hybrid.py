"""Ablation: can hybrid (prompt pool x local perturbation) sampling beat the
margin ceiling of single-prompt perturbation sampling?

Margin = similarity(positive real-SFT pair) - similarity(negative cross-family
pair): SmolLM2-135M vs SmolLM2-135M-Instruct (positive) and
SmolLM2-135M vs Qwen2.5-0.5B (negative). Reference: single-prompt
perturbation sampling gives margin ~0.13 (L=4, N=8), and the layer/sample
ablation showed the ceiling of that family is ~0.18.

All configs use L=4 layers, perturbation scale 2e-3, seed 123:
  single 1p x 8n  (N=8,  baseline)
  hybrid 4p x 2n  (N=8)
  hybrid 4p x 4n  (N=16)
  hybrid 8p x 2n  (N=16)

Usage:
    HF_ENDPOINT=https://hf-mirror.com python -m llm_transfer.ablate_hybrid
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffip import DiffIPMetric
from llm_transfer.llm import LLMRepresentationCollector

PROMPT = "The quick brown fox jumps over the lazy dog."

PROMPT_POOL_4 = (
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "The scientist carefully recorded the results of the experiment.",
    "Clouds gathered over the quiet mountain village at dusk.",
)
PROMPT_POOL_8 = PROMPT_POOL_4 + (
    "She brewed a cup of tea and opened her favorite book.",
    "The engine hummed steadily as the train left the station.",
    "Children played football in the park until sunset.",
    "Ancient trees line the narrow path to the old temple.",
)

SCALE = 2e-3
SEED = 123
N_COMPONENTS = 20
LAYERS = [0, 9, 19, 29]
QWEN_LAYERS = [0, 7, 15, 23]

VICTIM_ID = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
QWEN_ID = "Qwen/Qwen2.5-0.5B"

CONFIGS = [
    ("single 1p x 8n", dict(sample_mode="perturbation", n_samples=8)),
    ("hybrid 4p x 2n", dict(sample_mode="hybrid", prompt_pool=PROMPT_POOL_4, n_perturbations=2)),
    ("hybrid 4p x 4n", dict(sample_mode="hybrid", prompt_pool=PROMPT_POOL_4, n_perturbations=4)),
    ("hybrid 8p x 2n", dict(sample_mode="hybrid", prompt_pool=PROMPT_POOL_8, n_perturbations=2)),
]


def load(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    return model, tokenizer


def fingerprint(model_id: str, layers, config: dict):
    model, tokenizer = load(model_id)
    collector = LLMRepresentationCollector(
        model,
        tokenizer,
        perturbation_scale=SCALE,
        layers=layers,
        seed=SEED,
        **config,
    )
    representations = collector.collect(PROMPT).representations
    del model
    return representations


def main() -> None:
    metric = DiffIPMetric(n_components=N_COMPONENTS)
    print(f"{'config':>16} | {'positive':>8} | {'negative':>8} | {'margin':>7}", flush=True)
    print("-" * 50, flush=True)

    for name, config in CONFIGS:
        victim = fingerprint(VICTIM_ID, LAYERS, config)
        instruct = fingerprint(INSTRUCT_ID, LAYERS, config)
        qwen = fingerprint(QWEN_ID, QWEN_LAYERS, config)

        r_pos = metric.compare(instruct, victim)
        r_neg = metric.compare(qwen, victim)
        margin = r_pos.similarity - r_neg.similarity
        print(
            f"{name:>16} | {r_pos.similarity:>8.4f} | {r_neg.similarity:>8.4f} | {margin:>7.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
