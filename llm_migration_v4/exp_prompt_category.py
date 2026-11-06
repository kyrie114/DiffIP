"""Does prompt category explain the 0.956 fine_tune vs Qwen correlation?

Pre-registered before looking at the partial correlation. Categories are the
blocks in common.PROMPTS: English 0-6, Chinese 7-12, code 13-18, math 19-24,
QA 25-31. The raw correlation is computed again from scores/. The partial
correlation subtracts each prompt's category mean. Category explains the
shared prompt effect only if that partial correlation is below 0.5. At or
above 0.9, it does not. Between those, the result is inconclusive.
"""
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
SLICES = {
    "en": range(0, 7),
    "zh": range(7, 13),
    "code": range(13, 19),
    "math": range(19, 25),
    "qa": range(25, 32),
}


def prompt_means(name):
    rows = json.loads((H / "scores" / f"main_{name}.json").read_text())
    acc = {p: [] for p in range(32)}
    for row in rows:
        if row["N"] == 128 and row["dims"] == 8:
            acc[row["prompt"]].append(row["similarity"])
    return np.array([np.mean(acc[p]) for p in range(32)])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def demean(values):
    out = values.copy()
    for indexes in SLICES.values():
        idx = list(indexes)
        out[idx] = values[idx] - values[idx].mean()
    return out


def main():
    ft = prompt_means("fine_tune")
    qwen = prompt_means("neg_qwen2_5_7b")
    llama = prompt_means("neg_llama3_8b")
    raw_q = pearson(ft, qwen)
    raw_l = pearson(ft, llama)
    part_q = pearson(demean(ft), demean(qwen))
    part_l = pearson(demean(ft), demean(llama))
    if part_q < 0.5 and part_l < 0.5:
        verdict = "category explains the shared prompt effect"
    elif part_q >= 0.9 and part_l >= 0.9:
        verdict = "category does not explain the shared prompt effect"
    else:
        verdict = "inconclusive"
    by_cat = {}
    for name, indexes in SLICES.items():
        idx = list(indexes)
        by_cat[name] = {
            "n": len(idx),
            "ft_mean": float(ft[idx].mean()),
            "qwen_mean": float(qwen[idx].mean()),
            "within_ft_qwen": pearson(ft[idx], qwen[idx]) if len(idx) > 2 else None,
        }
    report = {
        "raw_ft_qwen": raw_q,
        "raw_ft_llama": raw_l,
        "partial_ft_qwen": part_q,
        "partial_ft_llama": part_l,
        "verdict": verdict,
        "by_category": by_cat,
    }
    (H / "exp_prompt_category.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
