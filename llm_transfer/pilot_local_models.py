"""ROC calibration over the local /data/luo_lixuan/llms model zoo.

Same protocol as roc_calibration.py (perturbation sampling, scale 2e-3,
seed 123, 8 prompts, per-prompt distances, 4 layers per model), but all
models come from the local disk — no HF download needed.

Pilot pairs:
  positive (real fine-tune lineage):
    Sheared-LLaMA-1.3B          <-> Sheared-LLaMA-1.3B-ShareGPT
  negatives:
    Llama-3.2-1B                <-> Sheared-LLaMA-1.3B        (different families)
    Llama-3.2-1B                <-> Llama-3.2-3B              (same family, different size)

Models are loaded one at a time (fp32, CPU) and fingerprinted per prompt;
each model is freed before the next loads.

Usage:
    conda run -n diffip-llm python -m llm_transfer.pilot_local_models
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffip import DiffIPMetric
from llm_transfer.llm import LLMRepresentationCollector

ROOT = Path("/data/luo_lixuan/llms")

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
N_LAYERS = 4

MODELS = (
    "Sheared-LLaMA-1.3B",
    "Sheared-LLaMA-1.3B-ShareGPT",
    "Llama-3.2-1B",
    "Llama-3.2-3B",
)

POSITIVE_PAIRS = [
    ("Sheared-LLaMA-1.3B-ShareGPT", "Sheared-LLaMA-1.3B"),
]
NEGATIVE_PAIRS = [
    ("Llama-3.2-1B", "Sheared-LLaMA-1.3B"),
    ("Llama-3.2-1B", "Llama-3.2-3B"),
]


def even_layers(total: int, count: int) -> list[int]:
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def fingerprint_per_prompt(name: str):
    path = ROOT / name
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    n_layers = model.config.num_hidden_layers
    layers = even_layers(n_layers, N_LAYERS)
    collector = LLMRepresentationCollector(
        model,
        tokenizer,
        n_samples=N_SAMPLES,
        perturbation_scale=SCALE,
        layers=layers,
        seed=SEED,
    )
    representations = {
        prompt: collector.collect(prompt).representations for prompt in PROMPTS
    }
    del model, collector
    gc.collect()
    return representations


def main() -> None:
    metric = DiffIPMetric(n_components=N_COMPONENTS)

    print(f"collecting per-prompt fingerprints for {len(MODELS)} local models ...",
          flush=True)
    fingerprints: dict[str, dict] = {}
    for name in MODELS:
        print(f"  {name} ...", flush=True)
        fingerprints[name] = fingerprint_per_prompt(name)
    print("done\n", flush=True)

    distances: list[float] = []
    labels: list[int] = []

    print(f"{'pair':<70} | per-prompt distances", flush=True)
    for suspect, victim in POSITIVE_PAIRS + NEGATIVE_PAIRS:
        label = 1 if (suspect, victim) in POSITIVE_PAIRS else 0
        pair_distances = [
            metric.compare(
                fingerprints[suspect][prompt], fingerprints[victim][prompt]
            ).distance
            for prompt in PROMPTS
        ]
        distances.extend(pair_distances)
        labels.extend([label] * len(pair_distances))
        pretty = " ".join(f"{d:6.2f}" for d in pair_distances)
        tag = "POS" if label else "NEG"
        print(f"({tag}) {suspect} <-> {victim}: {pretty}", flush=True)

    scores = [-d for d in distances]
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_index = tpr - fpr
    best = int(j_index.argmax())
    best_threshold = -thresholds[best]
    print(f"\nAUC = {auc:.4f}")
    print(
        f"best operating point (max Youden J={j_index[best]:.4f}): "
        f"threshold distance <= {best_threshold:.4f} -> same lineage "
        f"(TPR={tpr[best]:.3f}, FPR={fpr[best]:.3f})"
    )


if __name__ == "__main__":
    main()
