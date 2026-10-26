"""Validate the DiffIP LLM fingerprint with a REAL fine-tuned model pair.

Pairs (all evaluated with the calibrated protocol: perturbation sampling,
scale 2e-3, consistent noise seed):

  - self:    SmolLM2-135M vs itself                     (upper bound)
  - positive: SmolLM2-135M vs SmolLM2-135M-Instruct     (real SFT fine-tune,
             same lineage — expect HIGH similarity)
  - negative: SmolLM2-135M vs Qwen2.5-0.5B              (different family,
             different hidden size / layer count — expect LOW similarity)

The negative pair also exercises cross-dimension and unequal-depth support:
Qwen2.5-0.5B has 24 layers / D=896, SmolLM2-135M has 30 layers / D=576, and
the two models are sampled at different layer sets; the DP alignment handles
the unequal-length trajectories.

Usage:
    HF_ENDPOINT=https://hf-mirror.com python -m llm_transfer.validate_real_pair
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffip import DiffIPMetric
from llm_transfer.llm import LLMRepresentationCollector

PROMPT = "The quick brown fox jumps over the lazy dog."
N_SAMPLES = 8
PERTURBATION_SCALE = 2e-3
SEED = 123
N_COMPONENTS = 20

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
QWEN_ID = "Qwen/Qwen2.5-0.5B"


def load(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    return model, tokenizer


def fingerprint(model_id: str, layers) -> "tuple[str, object]":
    model, tokenizer = load(model_id)
    collector = LLMRepresentationCollector(
        model,
        tokenizer,
        n_samples=N_SAMPLES,
        perturbation_scale=PERTURBATION_SCALE,
        layers=layers,
        seed=SEED,
    )
    representations = collector.collect(PROMPT).representations
    print(f"  {model_id}: shape={representations.shape}", flush=True)
    del model
    return representations


def main() -> None:
    metric = DiffIPMetric(n_components=N_COMPONENTS)

    print("collecting fingerprints ...", flush=True)
    victim = fingerprint(MODEL_ID, layers=[0, 9, 19, 29])
    instruct = fingerprint(INSTRUCT_ID, layers=[0, 9, 19, 29])
    qwen = fingerprint(QWEN_ID, layers=[0, 7, 15, 23])

    print("\n== results ==", flush=True)
    r_self = metric.compare(victim, victim)
    r_pos = metric.compare(instruct, victim)
    r_neg = metric.compare(qwen, victim)
    print(f"self                (SmolLM2 vs SmolLM2)        similarity={r_self.similarity:.4f} distance={r_self.distance:.3e}")
    print(f"positive (real SFT) (SmolLM2 vs Instruct)       similarity={r_pos.similarity:.4f} distance={r_pos.distance:.3e}")
    print(f"negative (family)   (SmolLM2 vs Qwen2.5-0.5B)   similarity={r_neg.similarity:.4f} distance={r_neg.distance:.3e}")


if __name__ == "__main__":
    main()
