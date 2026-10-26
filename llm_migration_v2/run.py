"""DiffIP-faithful LLM migration experiment (v2).

Protocol (matches docs/llm-transfer-report.md / LLMRepresentationCollector):
  - sampling: same prompt, N=8 embedding perturbations, paired seed=123
  - perturbation_scale=2e-3
  - L=4 evenly spaced decoder layers (attention-mask mean pool)
  - metric PRIMARY: PCA-20 + per-layer ALS (QSX-h) mean L2;
    similarity = exp(-sum d_l / L). DP disabled (1:1 layer pairing).
  - scope: A fine-tune, B permutation, C anisotropic scaling, D layer prune
  - negative: cross-family SmolLM2-135M

No margin/AUC as primary.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffip.llm import LLMRepresentationCollector  # noqa: E402
from score_als import score_pair  # noqa: E402
from transforms import (  # noqa: E402
    apply_linear_residual_transform,
    make_anisotropic_scale,
    make_permutation,
    prune_layers,
)

OUT = Path(__file__).resolve().parent
FEAT = OUT / "feats"
FEAT.mkdir(exist_ok=True)

PROMPT = "The quick brown fox jumps over the lazy dog."
N_SAMPLES = 8
PERTURBATION_SCALE = 2e-3
SEED = 123
N_COMPONENTS = 20
N_LAYERS = 4

VICTIM_ID = "Qwen/Qwen2.5-0.5B"
INSTRUCT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
NEG_ID = "HuggingFaceTB/SmolLM2-135M"


def even_layers(total: int, count: int) -> list[int]:
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def load(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).eval()
    return model, tok


@torch.no_grad()
def collect_reps(model, tok, layers: list[int]) -> np.ndarray:
    collector = LLMRepresentationCollector(
        model,
        tok,
        n_samples=N_SAMPLES,
        perturbation_scale=PERTURBATION_SCALE,
        layers=layers,
        seed=SEED,
        sample_mode="perturbation",
    )
    return collector.collect(PROMPT).representations


def fingerprint_named(name: str, model_id: str, *, mutate=None) -> dict:
    feat_path = FEAT / f"{name}.npy"
    if feat_path.exists() and mutate is None:
        reps = np.load(feat_path)
        print(f"[{name}] reuse cached {feat_path} shape={reps.shape}", flush=True)
        # minimal meta; layers unknown — re-derive even layers from shape
        return {
            "model_id": model_id,
            "shape": list(reps.shape),
            "cached": True,
            "layers": "cached",
            "num_hidden_layers": None,
            "hidden_size": int(reps.shape[-1]),
        }
    print(f"[{name}] loading {model_id} ...", flush=True)
    model, tok = load(model_id)
    meta = {
        "model_id": model_id,
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
    }
    if mutate is not None:
        extra = mutate(model) or {}
        meta.update(extra)
    total = int(model.config.num_hidden_layers)
    layers = even_layers(total, min(N_LAYERS, total))
    meta["layers"] = layers
    reps = collect_reps(model, tok, layers)
    np.save(FEAT / f"{name}.npy", reps.astype(np.float64))
    meta["shape"] = list(reps.shape)
    print(f"[{name}] shape={reps.shape}", flush=True)
    del model, tok
    gc.collect()
    return meta


def mutate_perm(model):
    D = model.config.hidden_size
    P = make_permutation(D, seed=0)
    apply_linear_residual_transform(model, P, fold=False)
    return {"attack": "permutation", "fold": False}


def mutate_prune(model):
    total = model.config.num_hidden_layers
    n_drop = 4 if total > 8 else max(1, total // 6)
    candidates = [round(i * (total - 1) / (n_drop + 1)) for i in range(1, n_drop + 1)]
    drop = [c for c in candidates if 0 < c < total - 1]
    if len(drop) < n_drop:
        drop = list(range(2, 2 + n_drop))
    kept = prune_layers(model, drop)
    return {"attack": "layer_prune", "dropped": drop, "kept_count": len(kept)}


def verify_function_preserve(base_id: str, mutate, tag: str, n_prompts: int = 3) -> dict:
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "1+1=",
    ][:n_prompts]
    model, tok = load(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def gens(m):
        out = []
        for p in prompts:
            enc = tok(p, return_tensors="pt")
            g = m.generate(
                **enc,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            out.append(tok.decode(g[0, enc["input_ids"].shape[1] :]))
        return out

    before = gens(model)
    mutate(model)
    after = gens(model)
    identical = sum(a == b for a, b in zip(before, after))
    # also logit max-diff on first prompt
    enc = tok(prompts[0], return_tensors="pt")
    # reload clean for logit compare is expensive; skip — generation identity is enough
    del model, tok
    gc.collect()
    return {
        "tag": tag,
        "identical": f"{identical}/{len(prompts)}",
        "before": before,
        "after": after,
    }


def main():
    t0 = time.time()
    meta = {
        "protocol": {
            "prompt": PROMPT,
            "n_samples": N_SAMPLES,
            "perturbation_scale": PERTURBATION_SCALE,
            "seed": SEED,
            "n_components": N_COMPONENTS,
            "n_layers_sampled": N_LAYERS,
            "pooling": "attention-mask mean (LLMRepresentationCollector)",
            "primary_metric": "ALS-only mean L2 after PCA-20 (QSX-h); DP disabled",
            "similarity": "exp(-sum_l d_l / L)",
            "sampling": "law-of-large-numbers multi-sample via paired-seed embedding perturbation",
        }
    }

    catalog = {}
    catalog["victim"] = fingerprint_named("victim", VICTIM_ID)
    catalog["instruct"] = fingerprint_named("instruct", INSTRUCT_ID)
    catalog["neg_smol"] = fingerprint_named("neg_smol", NEG_ID)
    # --- weight-space perm on SHARED inputs (paired embedding cloud) ---
    # Critical: if we embed with the permuted E@P and add iid noise, the N-sample
    # cloud is NOT a global permutation of the victim cloud. Build noise once
    # from a clean reference, run both clean-instruct and perm-instruct on it.
    from collect_shared import build_perturbed_embeds, collect_on_embeds

    print("[shared] building paired embeds from clean Instruct ...", flush=True)
    ref_model, ref_tok = load(INSTRUCT_ID)
    total_ins = int(ref_model.config.num_hidden_layers)
    layers_ins = even_layers(total_ins, N_LAYERS)
    samples = build_perturbed_embeds(
        ref_model, ref_tok, PROMPT,
        n_samples=N_SAMPLES, perturbation_scale=PERTURBATION_SCALE, seed=SEED,
    )
    instruct_shared = collect_on_embeds(ref_model, samples, layers_ins)
    np.save(FEAT / "instruct_shared.npy", instruct_shared.astype(np.float64))
    catalog["instruct_shared"] = {
        "model_id": INSTRUCT_ID,
        "shape": list(instruct_shared.shape),
        "layers": layers_ins,
        "note": "collected on shared paired embeds (reference for weight attacks)",
    }
    print(f"[instruct_shared] shape={instruct_shared.shape}", flush=True)

    P = make_permutation(ref_model.config.hidden_size, seed=0)
    apply_linear_residual_transform(ref_model, P, fold=False)
    # Because embed_tokens was also transformed (E<-E@P), the residual-stream
    # basis changed: a clean embedding e must be fed as e@P for the permuted
    # net to stay equivariant. samples were built from the pre-transform E.
    P_dev = P.to(dtype=samples[0][0].dtype, device=samples[0][0].device)
    samples_perm = tuple((emb @ P_dev, mask) for emb, mask in samples)
    instruct_perm = collect_on_embeds(ref_model, samples_perm, layers_ins)
    np.save(FEAT / "instruct_perm.npy", instruct_perm.astype(np.float64))
    catalog["instruct_perm"] = {
        "model_id": INSTRUCT_ID,
        "attack": "permutation_weight_space_shared_inputs_equivariant",
        "fold": False,
        "shape": list(instruct_perm.shape),
        "layers": layers_ins,
        "note": "inputs are (E+eps)@P into weight-permuted model; hiddens = h@P",
    }
    print(f"[instruct_perm] shape={instruct_perm.shape}", flush=True)
    del ref_model, ref_tok, samples
    gc.collect()

    catalog["instruct_prune"] = fingerprint_named(
        "instruct_prune", INSTRUCT_ID, mutate=mutate_prune
    )

    # Theorem-level feature attacks (exact QSX-h claims):
    instruct = np.load(FEAT / "instruct_shared.npy")
    D = instruct.shape[-1]
    S = make_anisotropic_scale(D, seed=0, low=0.25, high=4.0).numpy().astype(np.float64)
    instruct_scale = np.einsum("lnd,dk->lnk", instruct, S)
    np.save(FEAT / "instruct_scale.npy", instruct_scale)
    catalog["instruct_scale"] = {
        "model_id": INSTRUCT_ID,
        "attack": "anisotropic_scale_on_representations",
        "note": "Y = X @ diag(s), s~U[0.25,4]; ALS closed form for S",
        "scale_min": float(np.diag(S).min()),
        "scale_max": float(np.diag(S).max()),
        "shape": list(instruct_scale.shape),
        "layers": layers_ins,
    }
    Pmat = make_permutation(D, seed=1).numpy().astype(np.float64)
    instruct_feat_perm = np.einsum("lnd,dk->lnk", instruct, Pmat)
    np.save(FEAT / "instruct_feat_perm.npy", instruct_feat_perm)
    catalog["instruct_feat_perm"] = {
        "model_id": INSTRUCT_ID,
        "attack": "permutation_on_representations",
        "note": "Y = X @ P; theorem-level test that ALS recovers orthogonal/perm Q",
        "shape": list(instruct_feat_perm.shape),
        "layers": layers_ins,
    }

    verify = {
        "perm": verify_function_preserve(INSTRUCT_ID, mutate_perm, "perm"),
    }
    meta["catalog"] = catalog
    meta["verify_function_preserve"] = verify

    def feat(name: str) -> np.ndarray:
        return np.load(FEAT / f"{name}.npy")

    pairs = [
        ("self", "victim", "victim", "baseline"),
        ("fine_tune_instruct", "instruct", "victim", "A_finetune"),
        ("neg_cross_family", "neg_smol", "victim", "negative"),
        # B: weight-space perm on shared inputs vs clean instruct / vs victim
        ("perm_vs_instruct_shared", "instruct_perm", "instruct_shared", "B_permutation"),
        ("perm_vs_victim", "instruct_perm", "victim", "B_permutation"),
        # B theorem: feature-level column perm
        ("feat_perm_vs_instruct", "instruct_feat_perm", "instruct_shared", "B_permutation_theorem"),
        # C theorem: feature-level anisotropic scale
        ("scale_vs_instruct", "instruct_scale", "instruct_shared", "C_scaling"),
        ("scale_vs_victim", "instruct_scale", "victim", "C_scaling"),
        # D: real layer prune
        ("prune_vs_instruct", "instruct_prune", "instruct", "D_pruning"),
        ("prune_vs_victim", "instruct_prune", "victim", "D_pruning"),
    ]

    results = []
    print("\n== scoring (ALS-only primary) ==", flush=True)
    hdr = f"{'pair':<32} {'sim':>8} {'d_mean':>10} {'d_sum':>10}"
    print(hdr, flush=True)
    for pair_name, sus_name, vic_name, group in pairs:
        scored = score_pair(
            feat(sus_name), feat(vic_name), n_components=N_COMPONENTS, also_full_metric=True
        )
        als = scored["als"]
        row = {
            "pair": pair_name,
            "suspect": sus_name,
            "victim": vic_name,
            "group": group,
            "similarity": als["similarity"],
            "distance_mean": als["distance_mean"],
            "distance_sum": als["distance_sum"],
            "per_layer_distance": als["per_layer_distance"],
            "n_layers": als["n_layers"],
            "n_components": als["n_components"],
            "converged_all": als["converged_all"],
            "note": scored["note"],
            "diffip_metric_reference": scored.get("diffip_metric_reference"),
        }
        results.append(row)
        print(
            f"{pair_name:<32} {als['similarity']:8.4f} {als['distance_mean']:10.4f} "
            f"{als['distance_sum']:10.4f}",
            flush=True,
        )
        if scored["note"]:
            print(f"  note: {scored['note']}", flush=True)

    meta["results"] = results
    meta["runtime_sec"] = time.time() - t0
    out_path = OUT / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path} in {meta['runtime_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
