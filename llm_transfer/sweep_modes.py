"""Compare fingerprint response curves for the two sampling modes.

For each sampling mode we sweep relative weight noise applied to a copy of
the victim model (0 = identical weights, up to 1e-1 = "unrelated" proxy) and
record DiffIP similarity. A wider, more stable within-model sample
distribution should turn the cliff-like response of perturbation sampling
into a usable separation between light and heavy noise.

Modes:
  - perturbation: 8 Gaussian-noise copies of one prompt (scale 2e-3, the
    operating point found by scripts/sweep_perturbation.py)
  - prompts:      8 distinct natural-sentence prompts (fixed pool)

Usage:
    HF_ENDPOINT=https://hf-mirror.com python -m llm_transfer.sweep_modes \
        --model HuggingFaceTB/SmolLM2-135M
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffip import DiffIPMetric
from llm_transfer.llm import LLMRepresentationCollector

PROMPT = "The quick brown fox jumps over the lazy dog."

PROMPT_POOL = (
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "The scientist carefully recorded the results of the experiment.",
    "Clouds gathered over the quiet mountain village at dusk.",
    "She brewed a cup of tea and opened her favorite book.",
    "The engine hummed steadily as the train left the station.",
    "Children played football in the park until sunset.",
    "Ancient trees line the narrow path to the old temple.",
)

NOISE_LEVELS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1)


def load_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def make_collector(model, tokenizer, mode: str) -> LLMRepresentationCollector:
    if mode == "perturbation":
        return LLMRepresentationCollector(
            model,
            tokenizer,
            n_samples=8,
            perturbation_scale=2e-3,
            layers=[0, 9, 19, 29],
            seed=123,
        )
    return LLMRepresentationCollector(
        model,
        tokenizer,
        sample_mode="prompts",
        prompt_pool=PROMPT_POOL,
        layers=[0, 9, 19, 29],
    )


def fingerprint(model_id: str, mode: str, rel_noise: float, seed: int):
    model, tokenizer = load_model(model_id)
    if rel_noise > 0:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(
                    torch.randn(parameter.shape, generator=generator)
                    * rel_noise
                    * parameter.abs().mean()
                )
    collector = make_collector(model, tokenizer, mode)
    representations = collector.collect(PROMPT).representations
    del model
    return representations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--n-components", type=int, default=20)
    args = parser.parse_args()

    metric = DiffIPMetric(n_components=args.n_components)

    for mode in ("perturbation", "prompts"):
        print(f"== sampling mode: {mode} ==", flush=True)
        victim = fingerprint(args.model, mode, 0.0, seed=0)
        r_self = metric.compare(victim, victim)
        print(f"{'rel noise':>10} | {'similarity':>10}", flush=True)
        print(f"{'self':>10} | {r_self.similarity:>10.4f}", flush=True)
        for rel_noise in NOISE_LEVELS[1:]:
            suspect = fingerprint(args.model, mode, rel_noise, seed=int(rel_noise * 1e6))
            result = metric.compare(suspect, victim)
            print(f"{rel_noise:>10.0e} | {result.similarity:>10.4f}", flush=True)


if __name__ == "__main__":
    main()
