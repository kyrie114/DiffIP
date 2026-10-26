"""PCA n_components sweep for DiffIP→LLM (llm_migration_v2 protocol).

Rank facts (official project_sequences):
  PCA is fit on flattened [L*N, D], so
    used = min(requested, L*N-1, D_suspect, D_victim).
  With the v2 working point N=8, L=4 → L*N-1 = 31.
  Requesting 20 is reachable; 32/48 are silently capped to 31.
  Per-layer ALS still sees only N points in the projected space.

This script:
  A) rescores existing N=8 feats across the grid (document used vs requested)
  B) recollects with N=64 so L*N-1=255, making 4..48 all rank-valid,
     then sweeps the same grid. N=64 is chosen ONLY for PCA-rank validity;
     the user's transfer report found N=16 already hurts separation — we
     report that honestly if it happens here too.

Primary metrics: ALS-only distance_mean and similarity (DP off).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V2))

from diffip.llm import LLMRepresentationCollector  # noqa: E402
from diffip.preprocessing import project_sequences  # noqa: E402
from score_als import als_only_compare  # noqa: E402
from transforms import make_anisotropic_scale, make_permutation  # noqa: E402

OUT = Path(__file__).resolve().parent
FEAT_N8 = V2 / "feats"
FEAT_N64 = OUT / "feats"
FEAT_N64.mkdir(exist_ok=True)

PROMPT = "The quick brown fox jumps over the lazy dog."
PERTURBATION_SCALE = 2e-3
SEED = 123
N_LAYERS = 4
GRID = [4, 8, 12, 16, 20, 32, 48]

VICTIM_ID = "Qwen/Qwen2.5-0.5B"
INSTRUCT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
NEG_ID = "HuggingFaceTB/SmolLM2-135M"


def even_layers(total: int, count: int) -> list[int]:
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def pca_rank_cap(L: int, N: int, D_s: int, D_v: int) -> int:
    return min(L * N - 1, D_s, D_v)


def used_components(requested: int, L: int, N: int, D_s: int, D_v: int) -> int:
    return min(requested, pca_rank_cap(L, N, D_s, D_v))


def load(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).eval()
    return model, tok


@torch.no_grad()
def collect_reps(model, tok, layers, n_samples: int) -> np.ndarray:
    collector = LLMRepresentationCollector(
        model,
        tok,
        n_samples=n_samples,
        perturbation_scale=PERTURBATION_SCALE,
        layers=layers,
        seed=SEED,
        sample_mode="perturbation",
    )
    return collector.collect(PROMPT).representations


def fingerprint(name: str, model_id: str, n_samples: int, feat_dir: Path) -> dict:
    path = feat_dir / f"{name}.npy"
    if path.exists():
        reps = np.load(path)
        print(f"[{name}] reuse {path} shape={reps.shape}", flush=True)
        return {
            "model_id": model_id,
            "shape": list(reps.shape),
            "cached": True,
            "hidden_size": int(reps.shape[-1]),
        }
    print(f"[{name}] loading {model_id} N={n_samples} ...", flush=True)
    model, tok = load(model_id)
    total = int(model.config.num_hidden_layers)
    layers = even_layers(total, min(N_LAYERS, total))
    reps = collect_reps(model, tok, layers, n_samples)
    np.save(path, reps.astype(np.float64))
    meta = {
        "model_id": model_id,
        "shape": list(reps.shape),
        "cached": False,
        "layers": layers,
        "num_hidden_layers": total,
        "hidden_size": int(model.config.hidden_size),
    }
    print(f"[{name}] shape={reps.shape}", flush=True)
    del model, tok
    gc.collect()
    return meta


def derive_attacks(instruct: np.ndarray, feat_dir: Path) -> dict:
    """Theorem-level feature attacks (cheap; no model reload)."""
    D = instruct.shape[-1]
    S = make_anisotropic_scale(D, seed=0, low=0.25, high=4.0).numpy().astype(np.float64)
    scale = np.einsum("lnd,dk->lnk", instruct, S)
    np.save(feat_dir / "instruct_scale.npy", scale)
    P = make_permutation(D, seed=1).numpy().astype(np.float64)
    feat_perm = np.einsum("lnd,dk->lnk", instruct, P)
    np.save(feat_dir / "instruct_feat_perm.npy", feat_perm)
    return {
        "instruct_scale": {
            "attack": "anisotropic_scale_on_representations",
            "shape": list(scale.shape),
            "scale_min": float(np.diag(S).min()),
            "scale_max": float(np.diag(S).max()),
        },
        "instruct_feat_perm": {
            "attack": "permutation_on_representations",
            "shape": list(feat_perm.shape),
        },
    }


def score_grid(pairs: list[tuple], feat_dir: Path, n_samples: int) -> list[dict]:
    rows = []
    for pair_name, sus_name, vic_name, group in pairs:
        sus = np.load(feat_dir / f"{sus_name}.npy")
        vic = np.load(feat_dir / f"{vic_name}.npy")
        L = min(sus.shape[0], vic.shape[0])
        N = sus.shape[1]
        assert N == n_samples, f"{sus_name} N={N} != expected {n_samples}"
        assert vic.shape[1] == n_samples
        D_s, D_v = sus.shape[-1], vic.shape[-1]
        cap = pca_rank_cap(L, N, D_s, D_v)
        for req in GRID:
            # probe official used count without scoring twice
            proj = project_sequences(sus[:L], vic[:L], req)
            used = int(proj.n_components)
            als = als_only_compare(sus[:L], vic[:L], n_components=req)
            row = {
                "regime": f"N={n_samples}",
                "pair": pair_name,
                "group": group,
                "suspect": sus_name,
                "victim": vic_name,
                "n_samples": n_samples,
                "n_layers": L,
                "D_suspect": D_s,
                "D_victim": D_v,
                "pca_rank_cap": cap,
                "n_components_requested": req,
                "n_components_used": used,
                "capped": used < req,
                "similarity": als.similarity,
                "distance_mean": als.distance_mean,
                "distance_sum": als.distance_sum,
                "per_layer_distance": als.per_layer_distance,
                "converged_all": als.converged_all,
            }
            rows.append(row)
            flag = " CAPPED" if used < req else ""
            print(
                f"N={n_samples} {pair_name:<28} req={req:>3} used={used:>3} "
                f"sim={als.similarity:.4f} d_mean={als.distance_mean:.4f}{flag}",
                flush=True,
            )
    return rows


def separation_table(rows: list[dict], n_samples: int) -> list[dict]:
    """Per requested dim: sim(fine_tune), sim(neg), gap = ft - neg."""
    by_req: dict[int, dict] = {}
    for r in rows:
        if r["n_samples"] != n_samples:
            continue
        req = r["n_components_requested"]
        by_req.setdefault(req, {"n_components_requested": req})
        by_req[req]["n_components_used_example"] = r["n_components_used"]
        by_req[req]["capped"] = r["capped"]
        by_req[req]["pca_rank_cap"] = r["pca_rank_cap"]
        if r["pair"] == "self":
            by_req[req]["sim_self"] = r["similarity"]
            by_req[req]["d_self"] = r["distance_mean"]
        elif r["pair"] == "fine_tune_instruct":
            by_req[req]["sim_finetune"] = r["similarity"]
            by_req[req]["d_finetune"] = r["distance_mean"]
        elif r["pair"] == "neg_cross_family":
            by_req[req]["sim_neg"] = r["similarity"]
            by_req[req]["d_neg"] = r["distance_mean"]
        elif r["pair"] == "feat_perm_vs_instruct":
            by_req[req]["sim_feat_perm"] = r["similarity"]
            by_req[req]["d_feat_perm"] = r["distance_mean"]
        elif r["pair"] == "scale_vs_instruct":
            by_req[req]["sim_scale"] = r["similarity"]
            by_req[req]["d_scale"] = r["distance_mean"]
    out = []
    for req in GRID:
        s = by_req.get(req, {"n_components_requested": req})
        if "sim_finetune" in s and "sim_neg" in s:
            s["gap_sim_ft_minus_neg"] = s["sim_finetune"] - s["sim_neg"]
            s["gap_d_neg_minus_ft"] = s["d_neg"] - s["d_finetune"]
        out.append(s)
    return out


def main():
    t0 = time.time()
    meta: dict = {
        "protocol": {
            "prompt": PROMPT,
            "perturbation_scale": PERTURBATION_SCALE,
            "seed": SEED,
            "n_layers_sampled": N_LAYERS,
            "pooling": "attention-mask mean (LLMRepresentationCollector)",
            "sample_mode": "perturbation, paired seed",
            "primary_metric": "ALS-only mean L2 after PCA; DP disabled (1:1)",
            "similarity": "exp(-sum_l d_l / L)",
            "grid": GRID,
            "rank_note": (
                "official project_sequences fits PCA on reshape(L*N, D); "
                "used = min(requested, L*N-1, D_s, D_v). "
                "N=8,L=4 → cap 31 so req 32/48 collapse to 31. "
                "N=64,L=4 → cap 255, grid fully valid."
            ),
        }
    }

    pairs = [
        ("self", "victim", "victim", "baseline"),
        ("fine_tune_instruct", "instruct", "victim", "A_finetune"),
        ("neg_cross_family", "neg_smol", "victim", "negative"),
        ("feat_perm_vs_instruct", "instruct_feat_perm", "instruct", "B_permutation_theorem"),
        ("scale_vs_instruct", "instruct_scale", "instruct", "C_scaling"),
    ]

    # ---------- Regime A: existing N=8 ----------
    print("\n== Regime A: N=8 (v2 working point; PCA cap=31) ==", flush=True)
    # ensure theorem attacks exist next to instruct (they do in v2 feats)
    if not (FEAT_N8 / "instruct_feat_perm.npy").exists():
        derive_attacks(np.load(FEAT_N8 / "instruct.npy"), FEAT_N8)
    # for N=8, feat_perm/scale were built from instruct_shared; use those names
    # Map: instruct_feat_perm vs instruct_shared in v2; we also have instruct.npy
    # Prefer instruct_shared if present for attack refs; else instruct
    pairs_n8 = list(pairs)
    if (FEAT_N8 / "instruct_shared.npy").exists():
        # rewrite attack pairs to use shared reference (matches v2 report)
        pairs_n8 = [
            ("self", "victim", "victim", "baseline"),
            ("fine_tune_instruct", "instruct", "victim", "A_finetune"),
            ("neg_cross_family", "neg_smol", "victim", "negative"),
            ("feat_perm_vs_instruct", "instruct_feat_perm", "instruct_shared", "B_permutation_theorem"),
            ("scale_vs_instruct", "instruct_scale", "instruct_shared", "C_scaling"),
        ]
    rows_n8 = score_grid(pairs_n8, FEAT_N8, n_samples=8)
    sep_n8 = separation_table(rows_n8, 8)

    # ---------- Regime B: N=64 for rank-valid high dims ----------
    print("\n== Regime B: N=64 (PCA cap=255; grid fully valid) ==", flush=True)
    catalog = {}
    catalog["victim"] = fingerprint("victim", VICTIM_ID, 64, FEAT_N64)
    catalog["instruct"] = fingerprint("instruct", INSTRUCT_ID, 64, FEAT_N64)
    catalog["neg_smol"] = fingerprint("neg_smol", NEG_ID, 64, FEAT_N64)
    attacks = derive_attacks(np.load(FEAT_N64 / "instruct.npy"), FEAT_N64)
    catalog.update(attacks)
    meta["catalog_N64"] = catalog

    pairs_n64 = [
        ("self", "victim", "victim", "baseline"),
        ("fine_tune_instruct", "instruct", "victim", "A_finetune"),
        ("neg_cross_family", "neg_smol", "victim", "negative"),
        ("feat_perm_vs_instruct", "instruct_feat_perm", "instruct", "B_permutation_theorem"),
        ("scale_vs_instruct", "instruct_scale", "instruct", "C_scaling"),
    ]
    rows_n64 = score_grid(pairs_n64, FEAT_N64, n_samples=64)
    sep_n64 = separation_table(rows_n64, 64)

    # recommend: among uncapped dims in the regime that matches protocol intent
    # Prefer N=8 (working point) among dims that are NOT capped; maximize gap_d
    def pick(sep_rows, require_uncapped: bool):
        cands = []
        for s in sep_rows:
            if "gap_d_neg_minus_ft" not in s:
                continue
            if require_uncapped and s.get("capped"):
                continue
            cands.append(s)
        if not cands:
            return None
        # maximize distance gap; tie-break: keep self≈1 and perm/scale recovery high
        cands.sort(
            key=lambda s: (
                s["gap_d_neg_minus_ft"],
                s.get("sim_feat_perm", 0) + s.get("sim_scale", 0),
                -s["n_components_requested"],  # prefer smaller if tied
            ),
            reverse=True,
        )
        return cands[0]

    rec_n8 = pick(sep_n8, require_uncapped=True)
    rec_n64 = pick(sep_n64, require_uncapped=True)

    meta["results"] = rows_n8 + rows_n64
    meta["separation_N8"] = sep_n8
    meta["separation_N64"] = sep_n64
    meta["recommendation"] = {
        "N8_working_point": {
            "note": "protocol N=8; only dims with used==requested (≤31)",
            "best": rec_n8,
        },
        "N64_rank_valid": {
            "note": (
                "N raised only so PCA rank allows 48; "
                "prior ablate found larger N can collapse positives — check gaps"
            ),
            "best": rec_n64,
        },
    }
    meta["runtime_sec"] = time.time() - t0
    out_path = OUT / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path} in {meta['runtime_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
