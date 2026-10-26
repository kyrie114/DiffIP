"""Per-prompt scoring + aggregation (the fix for the hybrid-sampling failure).

Hybrid sampling collapsed because mixing heterogeneous prompts into one
sample cloud violates the paired-Procrustes assumption (both clouds must
differ by a single linear map). Here each prompt is scored SEPARATELY with
the standard perturbation protocol (homogeneous cloud per prompt), and the
per-prompt distances are aggregated (mean / median) before conversion to
similarity: sim = exp(-aggregated_distance / sqrt(T_s * T_v)).

Pairs: SmolLM2-135M vs SmolLM2-135M-Instruct (positive) and SmolLM2-135M vs
Qwen2.5-0.5B (negative). Reference margin (single prompt): 0.182.

Usage:
    HF_ENDPOINT=https://hf-mirror.com python -m llm_transfer.per_prompt_eval
"""

from __future__ import annotations

import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffip import DiffIPMetric
from llm_transfer.llm import LLMRepresentationCollector

PROMPTS = (
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "The scientist carefully recorded the results of the experiment.",
    "Clouds gathered over the quiet mountain village at dusk.",
    "She brewed a cup of tea and opened her favorite book.",
    "The engine hummed steadily as the train left the station.",
    "Children played football in the park until sunset.",
    "Ancient trees line the narrow path to the old temple.",
)

SCALE = 2e-3
SEED = 123
N_SAMPLES = 8
N_COMPONENTS = 20
LAYERS = [0, 9, 19, 29]
QWEN_LAYERS = [0, 7, 15, 23]

VICTIM_ID = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
QWEN_ID = "Qwen/Qwen2.5-0.5B"


def load(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    return model, tokenizer


def main() -> None:
    metric = DiffIPMetric(n_components=N_COMPONENTS)

    def fingerprint_per_prompt(model_id: str, layers):
        model, tokenizer = load(model_id)
        collector = LLMRepresentationCollector(
            model,
            tokenizer,
            n_samples=N_SAMPLES,
            perturbation_scale=SCALE,
            layers=layers,
            seed=SEED,
        )
        reps = {prompt: collector.collect(prompt).representations for prompt in PROMPTS}
        del model
        return reps

    print("collecting per-prompt fingerprints ...", flush=True)
    victim = fingerprint_per_prompt(VICTIM_ID, LAYERS)
    instruct = fingerprint_per_prompt(INSTRUCT_ID, LAYERS)
    qwen = fingerprint_per_prompt(QWEN_ID, QWEN_LAYERS)

    def score_pair(suspect_reps, victim_reps, n_prompts: int):
        """Aggregate per-prompt distances over the first n_prompts prompts."""
        distances = np.array(
            [
                metric.compare(suspect_reps[p], victim_reps[p]).distance
                for p in PROMPTS[:n_prompts]
            ]
        )
        normalization = math.sqrt(
            len(LAYERS) * N_SAMPLES * len(LAYERS) * N_SAMPLES
        )
        mean_sim = math.exp(-distances.mean() / normalization)
        median_sim = math.exp(-np.median(distances) / normalization)
        return distances, mean_sim, median_sim

    for n_prompts in (1, 4, 8):
        print(f"\n== aggregating over {n_prompts} prompt(s) ==", flush=True)
        d_pos, mean_pos, med_pos = score_pair(instruct, victim, n_prompts)
        d_neg, mean_neg, med_neg = score_pair(qwen, victim, n_prompts)
        print("per-prompt distances (positive):", np.array2string(d_pos, precision=3), flush=True)
        print("per-prompt distances (negative):", np.array2string(d_neg, precision=3), flush=True)
        print(
            f"mean   similarity: positive={mean_pos:.4f} negative={mean_neg:.4f} "
            f"margin={mean_pos - mean_neg:+.4f}",
            flush=True,
        )
        print(
            f"median similarity: positive={med_pos:.4f} negative={med_neg:.4f} "
            f"margin={med_pos - med_neg:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
