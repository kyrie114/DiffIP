"""Sweep input-perturbation scale to find a usable DiffIP operating point.

For each perturbation scale we compare the victim fingerprint against:
  - itself                (upper bound, expect similarity ~ 1)
  - light weight noise    (rel 3e-4, simulates a light fine-tune = target signal)
  - heavy weight noise    (rel 1e-1, negative proxy for an unrelated model)

A good operating point shows a wide separation between the light-noise score
and the heavy-noise score, plus a self score close to 1.

Usage:
    HF_ENDPOINT=https://hf-mirror.com python -m scripts.sweep_perturbation \
        --model HuggingFaceTB/SmolLM2-135M
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffip import DiffIPMetric
from diffip.llm import LLMRepresentationCollector


def load_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def fingerprint_with_noise(model_id: str, rel_noise: float, seed: int, n_samples: int, layers):
    """Fingerprint a copy of the model with relative weight noise applied."""
    model, tokenizer = load_model(model_id)
    generator = torch.Generator().manual_seed(seed)
    if rel_noise > 0:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(
                    torch.randn(parameter.shape, generator=generator)
                    * rel_noise
                    * parameter.abs().mean()
                )
    collector = LLMRepresentationCollector(
        model, tokenizer, n_samples=n_samples, layers=layers
    )
    representations = collector.collect(PROMPT).representations
    del model
    return representations


PROMPT = "The quick brown fox jumps over the lazy dog."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--layers", default="0,9,19,29")
    parser.add_argument(
        "--scales", default="1e-3,1e-2,1e-1,0.5",
        help="comma-separated perturbation scales to sweep",
    )
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    scales = [float(x) for x in args.scales.split(",")]
    metric = DiffIPMetric(n_components=args.n_components)

    print(f"loading victim: {args.model}")
    victim_model, victim_tokenizer = load_model(args.model)
    victim_collector = LLMRepresentationCollector(
        victim_model, victim_tokenizer, n_samples=args.n_samples, layers=layers
    )

    header = f"{'scale':>8} | {'self':>8} | {'light 3e-4':>10} | {'heavy 1e-1':>10}"
    print(header)
    print("-" * len(header))

    for scale in scales:
        victim_collector.perturbation_scale = scale
        victim = victim_collector.collect(PROMPT).representations

        light = fingerprint_with_noise(args.model, 3e-4, seed=1, n_samples=args.n_samples, layers=layers)
        heavy = fingerprint_with_noise(args.model, 1e-1, seed=2, n_samples=args.n_samples, layers=layers)

        r_self = metric.compare(victim, victim)
        r_light = metric.compare(light, victim)
        r_heavy = metric.compare(heavy, victim)
        print(
            f"{scale:>8.0e} | {r_self.similarity:>8.4f} | "
            f"{r_light.similarity:>10.4f} | {r_heavy.similarity:>10.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
