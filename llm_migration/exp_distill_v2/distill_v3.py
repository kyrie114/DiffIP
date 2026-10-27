"""Distill line rerun under the llm_migration_v3 protocol (scheme a).

Question: can the DiffIP-faithful fingerprint (paired perturbation seeds,
per-prompt scoring, ALS reversion, no DP) recognise a sequence-level
distilled student, where v1 (margin protocol, small config) said no?

Victim = SmolLM2-135M (the base the students are derived from).
Suspects: student_distill (trained on teacher answers), student_control
(same instruction/budget on raw Alpaca answers — the critical control),
plus the standard negative panel.

Protocol identical to llm_migration_v3: SCALE 2e-3, paired seed sets
{123,1123,2123}, 32 prompts, N in {8,16,20,32,64,128}, dims {8,20},
4 even layers, attention-mask mean pooling, PCA + per-layer ALS,
similarity = exp(-sum_l d_l / L).

Usage (after ckpt/student_distill and ckpt/student_control exist):
    cd llm_migration/exp_distill_v2 && python distill_v3.py collect
    cd llm_migration/exp_distill_v2 && python distill_v3.py score
"""

from __future__ import annotations

import gc
import itertools
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "llm_migration_v3"))

from common import (  # noqa: E402
    N_GRID, PROMPTS, SCALE, SEED_SETS, even_layers, forward_pool, load,
    perturb_batch,
)
from diffip.preprocessing import project_sequences  # noqa: E402
from diffip.reversion import fit_step_alignment  # noqa: E402

H = Path(__file__).resolve().parent
FEATS = H / "feats_v3"
VICTIM = "HuggingFaceTB/SmolLM2-135M"
SUSPECTS = {
    "student_distill": "ckpt/student_distill",
    "student_control": "ckpt/student_control",
    "neg_smollm2_360m": "HuggingFaceTB/SmolLM2-360M",
    "neg_gpt2": "openai-community/gpt2",
    "neg_pythia_160m": "EleutherAI/pythia-160m",
    "neg_qwen2_0.5b": "Qwen/Qwen2-0.5B",
}
DIMS = [8, 20]
GRID = sorted(set(N_GRID + [20]))
N_LAYERS = 4


def collect() -> None:
    FEATS.mkdir(exist_ok=True)
    names = ["victim", *SUSPECTS]
    for name in names:
        out = FEATS / f"{name}.npz"
        if out.exists():
            print("skip", name, flush=True)
            continue
        path = VICTIM if name == "victim" else H / SUSPECTS[name]
        print(f"loading {name} from {path} ...", flush=True)
        model, tok = load(str(path))
        layers = even_layers(model.config.num_hidden_layers, N_LAYERS)
        feats = np.zeros(
            (len(PROMPTS), len(SEED_SETS), N_LAYERS, max(N_GRID),
             model.config.hidden_size), dtype=np.float32
        )
        for pi, prompt in enumerate(PROMPTS):
            ids = tok(prompt, return_tensors="pt").input_ids
            base = model.get_input_embeddings()(ids)[0]
            for ki, seed in enumerate(SEED_SETS):
                embeds = perturb_batch(base, max(N_GRID), seed)
                pooled = pooled_layers(model, embeds, layers)
                feats[pi, ki] = pooled.transpose(1, 0, 2)
            print(f"  {name} prompt {pi + 1}/{len(PROMPTS)}", flush=True)
        np.savez_compressed(out, pert=feats)
        del model
        gc.collect()
        print("saved", out, flush=True)


def pooled_layers(model, embeds, layers):
    """[n, T, D] -> [len(layers), n, D] via output_hidden_states (no hooks).

    Falls back to batch=1 if the batched forward fails (transformers-version
    dependent RoPE bug observed once with batched inputs_embeds)."""
    am = torch.ones(embeds.shape[:2], dtype=torch.long)
    try:
        with torch.no_grad():
            out = model(inputs_embeds=embeds, attention_mask=am,
                        use_cache=False, output_hidden_states=True)
        hs = out.hidden_states  # [embed, layer0, ..., layer_{L-1}]
        picked = torch.stack([hs[i + 1] for i in layers], 0)  # [L, n, T, D]
        return picked.mean(dim=2).numpy()
    except RuntimeError as e:
        print(f"  batched forward failed ({str(e)[:60]}); falling back to batch=1",
              flush=True)
        res = [[] for _ in layers]
        for s in range(embeds.shape[0]):
            with torch.no_grad():
                out = model(inputs_embeds=embeds[s:s + 1],
                            attention_mask=am[s:s + 1],
                            use_cache=False, output_hidden_states=True)
            hs = out.hidden_states
            for k, i in enumerate(layers):
                res[k].append(hs[i + 1][0].mean(dim=0))
        return torch.stack([torch.stack(r, 0) for r in res], 0).numpy()


def als(sus, vic, d):
    pr = project_sequences(sus, vic, d)
    dl = [fit_step_alignment(pr.suspect[i], pr.victim[i]).distance
          for i in range(sus.shape[0])]
    return float(np.mean(dl)), math.exp(-sum(dl) / len(dl)), pr.n_components


def _job(a):
    name, pi, ki, n, d = a
    assert d <= 4 * n - 1
    sus = np.load(FEATS / f"{name}.npz")["pert"].astype(np.float64)
    vic = np.load(FEATS / "victim.npz")["pert"].astype(np.float64)
    dm, sim, _ = als(sus[pi, ki, :, :n], vic[pi, ki, :, :n], d)
    return dict(pair=name, prompt=pi, seed_set=ki, N=n, dims=d,
                distance_mean=dm, similarity=sim)


def score() -> None:
    from multiprocessing import Pool
    rows = []
    jobs = [itertools.product([name], range(len(PROMPTS)),
                              range(len(SEED_SETS)), GRID, DIMS)
            for name in SUSPECTS]
    jobs = list(itertools.chain(*jobs))
    with Pool(8) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=16)):
            rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    out = H / "scores_v3.json"
    out.write_text(json.dumps(rows))
    print("scored", len(rows), "total", flush=True)
    summarize(rows)


def summarize(rows: list[dict]) -> None:
    import collections
    agg = collections.defaultdict(dict)
    for r in rows:
        key = (r["pair"], r["N"], r["dims"])
        agg[key].setdefault(r["seed_set"], []).append(r["similarity"])
    print("\n== 跨 32 prompt 均值的 similarity（3 个 seed set 的 mean ± std）==")
    pairs = sorted({k[0] for k in agg})
    for dims in DIMS:
        print(f"\n-- dims={dims} --")
        print(f"{'pair':<20}" + "".join(f"{n:>14}" for n in GRID))
        for pair in pairs:
            line = f"{pair:<20}"
            for n in GRID:
                per_seed = [np.mean(v) for (p, nn, dd), v in agg.items()
                            if p == pair and nn == n and dd == dims]
                line += f"{np.mean(per_seed):>9.3f}±{np.std(per_seed):<4.3f}"
            print(line)
    # per-prompt win counts at N=20 and N=128
    for n in (20, max(N_GRID)):
        for dims in DIMS:
            print(f"\n== 逐 prompt 胜负（N={n}, dims={dims}, seed set 0）："
                  f"该 suspect 高于同 prompt 全部负例的 prompt 数 / 32 ==")
            by = collections.defaultdict(lambda: np.zeros(len(PROMPTS)))
            for r in rows:
                if r["seed_set"] == 0 and r["N"] == n and r["dims"] == dims:
                    by[r["pair"]][r["prompt"]] = r["similarity"]
            negs = np.max(np.stack([by[k] for k in by if k.startswith("neg_")]),
                          axis=0)
            for pair in ("student_distill", "student_control"):
                wins = int((by[pair] > negs).sum())
                print(f"  {pair}: {wins}/32   "
                      f"sim mean={by[pair].mean():.3f} min={by[pair].min():.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "collect"
    if cmd == "collect":
        collect()
    elif cmd == "score":
        score()
    else:
        raise SystemExit(f"unknown command {cmd!r} (use collect|score)")
