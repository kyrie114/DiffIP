"""Official REEF activations for the 2026-11-11 validity/robustness plan.

TruthfulQA first 200 rows, file order, no shuffle. Statement is
Question.strip() + " " + Best Answer.strip(). No chat template, no truncation.
Last-token hidden state of every decoder layer (hidden_states[i+1]).
One fresh process per model. Output: acts_reef_tqa/{name}.npz
  last  float32 [200, layers, D]
  total int64
"""
from __future__ import annotations

import csv
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from collect import n_decoder_layers
from common import MAIN, PANEL, UNRELATED_ROUND, build_main

H = Path(__file__).resolve().parent
OUT = H / "acts_reef_tqa"
TQA_CSV = H.parent / "llm_migration_v3" / "reef_baseline" / "TruthfulQA.csv"
N_STATEMENTS = 200
# Not the default no-arg DiffIP roster. internlm2, weight_perm, and prune_dpoff
# stay out of this round.
DEFAULT = [
    "victim", "fine_tune", "derivative_cn", "pos_vicuna",
    "neg_qwen2_5_7b", "neg_llama3_8b", "neg_llama1_7b", "neg_baichuan_7b", "neg_mpt_7b",
    "neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm",
    "pos_sheared_1_3b_pruned", "pos_sheared_1_3b", "pos_sheared_1_3b_sharegpt",
    "pos_sheared_2_7b_pruned", "pos_sheared_2_7b", "pos_sheared_2_7b_sharegpt",
    "cpt_codellama", "cpt_wizardmath", "cpt_llemma",
]
REFUSED = {"neg_internlm2_7b", "weight_perm", "prune_dpoff", "rep_perm", "scale"}


def statements():
    rows = list(csv.DictReader(open(TQA_CSV, newline="", encoding="utf-8")))
    if len(rows) < N_STATEMENTS:
        raise SystemExit(f"TruthfulQA has {len(rows)} rows, need {N_STATEMENTS}")
    need = {"Question", "Best Answer"}
    if not need <= set(rows[0]):
        raise SystemExit(f"TruthfulQA columns missing {need - set(rows[0])}")
    return [r["Question"].strip() + " " + r["Best Answer"].strip()
            for r in rows[:N_STATEMENTS]]


def cache_ok(path):
    if not path.exists():
        return False
    z = np.load(path)
    last = z["last"]
    total = int(z["total"])
    return (last.ndim == 3 and last.shape[0] == N_STATEMENTS
            and last.shape[1] == total and np.isfinite(last).all())


def collect_one(name, texts):
    import torch
    torch.set_num_threads(int(os.environ.get("V4_THREADS", "8")))
    out = OUT / f"{name}.npz"
    if out.exists() and not cache_ok(out):
        raise SystemExit(f"bad cache {name}: delete acts_reef_tqa/{name}.npz and rerun")
    if cache_ok(out):
        print("skip", name, flush=True)
        return
    t0 = time.time()
    model, tok, emb, perm_idx = build_main(name)
    total = n_decoder_layers(model)
    D = model.get_input_embeddings().weight.shape[1]
    last = np.zeros((len(texts), total, D), dtype=np.float32)
    for p, text in enumerate(texts):
        ids = tok(text, return_tensors="pt").input_ids
        x = emb(ids).detach()
        if perm_idx is not None:
            x = x[..., perm_idx]
        am = torch.ones(x.shape[:2], dtype=torch.long)
        with torch.no_grad():
            hs = model(inputs_embeds=x, attention_mask=am, use_cache=False,
                       output_hidden_states=True, return_dict=True).hidden_states
        if len(hs) != total + 1:
            raise SystemExit(f"{name}: hidden_states {len(hs)} != layers+1 {total + 1}")
        for i in range(total):
            last[p, i] = hs[i + 1][0, -1].float().cpu().numpy()
        if not np.isfinite(last[p]).all():
            bad = last[p]
            raise SystemExit(
                f"{name} statement {p + 1}: non-finite last token "
                f"nan={int(np.isnan(bad).sum())} inf={int(np.isinf(bad).sum())}")
        if (p + 1) % 20 == 0 or p + 1 == len(texts):
            print(f"  {name} {p + 1}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(exist_ok=True)
    tmp = OUT / f"{name}.writing.npz"
    np.savez(str(tmp), last=last, total=np.int64(total))
    os.replace(tmp, out)
    print(f"saved {name} layers={total} D={D} {time.time() - t0:.0f}s", flush=True)
    del model
    gc.collect()


def main():
    names = sys.argv[1:] or DEFAULT
    known = set(MAIN) | set(UNRELATED_ROUND) | set(PANEL)
    unknown = [n for n in names if n not in known or n in REFUSED]
    if unknown:
        raise SystemExit(f"refused or unknown: {unknown}")
    if os.environ.get("V4_REEF_TQA_WORKER") == "1":
        texts = statements()
        for name in names:
            collect_one(name, texts)
        return
    OUT.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["V4_REEF_TQA_WORKER"] = "1"
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    for name in names:
        path = OUT / f"{name}.npz"
        if cache_ok(path):
            print("skip", name, flush=True)
            continue
        if path.exists():
            raise SystemExit(f"bad cache {name}: delete acts_reef_tqa/{name}.npz and rerun")
        proc = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), name], env=env)
        if proc.returncode != 0:
            raise SystemExit(f"{name} exited {proc.returncode}")


if __name__ == "__main__":
    main()
