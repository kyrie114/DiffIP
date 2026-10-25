"""ROC calibration of the DiffIP LLM fingerprint over real model pairs.

Pairs (protocol: perturbation sampling, scale 2e-3, seed 123, 8 prompts,
per-prompt distances; 4 layers per model):

  positives (real SFT lineage):
    SmolLM2-135M      <-> SmolLM2-135M-Instruct
    Qwen2.5-0.5B      <-> Qwen2.5-0.5B-Instruct
  negatives (different families):
    all SmolLM2 <-> Qwen2.5-0.5B / Qwen2.5-0.5B-Instruct combinations

Each pair contributes 8 per-prompt distance samples (16 positive, 32
negative). We sweep the distance threshold and report the best operating
point by Youden's J, plus the AUC. Distances are negated before AUC so that
"more similar" (smaller distance) counts towards the positive class.

Usage:
    HF_ENDPOINT=https://hf-mirror.com python -m scripts.roc_calibration
"""

from __future__ import annotations

import torch
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffip import DiffIPMetric
from diffip.llm import LLMRepresentationCollector

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

SMOL_ID = "HuggingFaceTB/SmolLM2-135M"
SMOL_INST_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
QWEN_ID = "Qwen/Qwen2.5-0.5B"
QWEN_INST_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SMOL_LAYERS = [0, 9, 19, 29]
QWEN_LAYERS = [0, 7, 15, 23]

POSITIVE_PAIRS = [
    (SMOL_INST_ID, SMOL_ID),
    (QWEN_INST_ID, QWEN_ID),
]
NEGATIVE_PAIRS = [
    (SMOL_ID, QWEN_ID),
    (SMOL_ID, QWEN_INST_ID),
    (SMOL_INST_ID, QWEN_ID),
    (SMOL_INST_ID, QWEN_INST_ID),
]

LAYERS_BY_MODEL = {
    SMOL_ID: SMOL_LAYERS,
    SMOL_INST_ID: SMOL_LAYERS,
    QWEN_ID: QWEN_LAYERS,
    QWEN_INST_ID: QWEN_LAYERS,
}


def fingerprint_per_prompt(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    collector = LLMRepresentationCollector(
        model,
        tokenizer,
        n_samples=N_SAMPLES,
        perturbation_scale=SCALE,
        layers=LAYERS_BY_MODEL[model_id],
        seed=SEED,
    )
    representations = {
        prompt: collector.collect(prompt).representations for prompt in PROMPTS
    }
    del model
    return representations


def main() -> None:
    metric = DiffIPMetric(n_components=N_COMPONENTS)

    print("collecting per-prompt fingerprints for 4 models ...", flush=True)
    fingerprints = {model_id: fingerprint_per_prompt(model_id) for model_id in
                    (SMOL_ID, SMOL_INST_ID, QWEN_ID, QWEN_INST_ID)}
    print("done\n", flush=True)

    distances: list[float] = []
    labels: list[int] = []

    print(f"{'pair':<52} | per-prompt distances", flush=True)
    for suspect_id, victim_id in POSITIVE_PAIRS + NEGATIVE_PAIRS:
        label = 1 if (suspect_id, victim_id) in POSITIVE_PAIRS else 0
        pair_distances = [
            metric.compare(
                fingerprints[suspect_id][prompt], fingerprints[victim_id][prompt]
            ).distance
            for prompt in PROMPTS
        ]
        distances.extend(pair_distances)
        labels.extend([label] * len(pair_distances))
        pretty = " ".join(f"{d:5.2f}" for d in pair_distances)
        tag = "POS" if label else "NEG"
        print(f"({tag}) {suspect_id} <-> {victim_id}: {pretty}", flush=True)

    scores = [-d for d in distances]  # negate: smaller distance = more similar
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_index = tpr - fpr
    best = int(j_index.argmax())
    best_threshold = -thresholds[best]  # back to distance scale
    print(f"\nAUC = {auc:.4f}")
    print(
        f"best operating point (max Youden J={j_index[best]:.4f}): "
        f"threshold distance <= {best_threshold:.4f} -> same lineage "
        f"(TPR={tpr[best]:.3f}, FPR={fpr[best]:.3f})"
    )


if __name__ == "__main__":
    main()
