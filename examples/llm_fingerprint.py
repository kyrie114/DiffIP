"""Skeleton: fingerprint an LLM with DiffIP via the transformers collector.

Usage:
    python -m examples.llm_fingerprint --model <hf-model-id> [--victim <hf-model-id>]

When only --model is given, the same model fingerprint is compared against a
perturbed copy of itself as a smoke test. Replace the victim with a
fine-tuned / distilled suspect model for a real lineage check.
"""

from __future__ import annotations

import argparse

from diffip import DiffIPMetric
from diffip.llm import LLMRepresentationCollector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="victim model id on the Hub")
    parser.add_argument("--victim", default=None, help="suspect model id (defaults to --model)")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--layers", default=None, help="comma-separated layer indices, e.g. 0,5,11")
    parser.add_argument("--perturbation-scale", type=float, default=2e-3)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument(
        "--seed", type=int, default=123,
        help="noise seed; must be consistent across the compared models",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load(model_id: str):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        model.eval()
        return model, tokenizer

    layers = (
        [int(x) for x in args.layers.split(",")] if args.layers else None
    )

    victim_model, victim_tokenizer = load(args.model)
    victim_collector = LLMRepresentationCollector(
        victim_model,
        victim_tokenizer,
        n_samples=args.n_samples,
        perturbation_scale=args.perturbation_scale,
        layers=layers,
        seed=args.seed,
    )
    victim = victim_collector.collect(args.prompt).representations
    print(f"victim representations: {victim.shape}")  # [L, N, D]

    if args.victim is None:
        suspect = victim  # smoke test: identical model, distance ~ 0
    else:
        suspect_model, suspect_tokenizer = load(args.victim)
        suspect_collector = LLMRepresentationCollector(
            suspect_model,
            suspect_tokenizer,
            n_samples=args.n_samples,
            perturbation_scale=args.perturbation_scale,
            layers=layers,
            seed=args.seed,
        )
        suspect = suspect_collector.collect(args.prompt).representations
        print(f"suspect representations: {suspect.shape}")

    result = DiffIPMetric(n_components=args.n_components).compare(suspect, victim)
    print("similarity:", result.similarity)
    print("distance:", result.distance)


if __name__ == "__main__":
    main()
