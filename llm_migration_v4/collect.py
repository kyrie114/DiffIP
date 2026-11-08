"""Collect fingerprint features for a line.

Usage:
    python collect.py main [names...]      # default: all MAIN models
    python collect.py distill <names...>   # names must exist (students trained)

Output: feats/{line}_{name}.npz with pert [32 prompts, 3 seed sets, 4 layers, 128, D],
layers4, total. Reproducible per-prompt fingerprints (v3 protocol scheme a).

V4_ALL_LAYERS=1 stores every decoder layer instead, under feats_all/. The
historical 4-layer files in feats/ are not overwritten. New models in the
维数/层数/无关模型 round are collected with this switch on.

Each finished prompt is also written to a partial npz so a killed run resumes
at the next prompt instead of redoing the model.
"""
import gc
import os
import sys
import time

import numpy as np

from common import (H, MAIN, DISTILL, N_LAYERS, N_MAX, PROMPTS, SEED_SETS,
                    UNRELATED_ROUND, build_main, even_layers, load, pooled_layers,
                    perturb_batch)

ALL_LAYERS = os.environ.get("V4_ALL_LAYERS", "0") == "1"
OUT = H / ("feats_all" if ALL_LAYERS else "feats")
OUT.mkdir(exist_ok=True)


def n_decoder_layers(model):
    """Llama/Qwen/Baichuan use model.layers. MPT uses transformer.blocks."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return len(inner.layers)
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "blocks"):
        return len(tr.blocks)
    raise SystemExit(f"no decoder stack on {type(model).__name__}")


def _save_partial(path, feats, L4, total, done):
    tmp = path.parent / (path.name + ".writing.npz")
    np.savez(tmp, pert=feats, layers4=np.array(L4), total=np.int64(total),
             done=np.int64(done))
    os.replace(tmp, path)


def collect(line, names):
    for name in names:
        out = OUT / f"{line}_{name}.npz"
        if out.exists():
            print("skip", name, flush=True)
            continue
        partial = OUT / f"{line}_{name}.partial.npz"
        t0 = time.time()
        if line == "main":
            m, tok, emb, perm_idx = build_main(name)
        else:
            from common import DISTILL as D
            m, tok = load(D[name])
            E = m.get_input_embeddings()
            emb, perm_idx = (lambda ids: E(ids)), None
        total = n_decoder_layers(m)
        # All-layer dumps keep absolute decoder indices. The 4-layer path
        # stays on even_layers, which is what the judged feats/ files used.
        L4 = list(range(total)) if ALL_LAYERS else even_layers(total, N_LAYERS)
        D_ = m.get_input_embeddings().weight.shape[1]
        shape = (len(PROMPTS), len(SEED_SETS), len(L4), N_MAX, D_)
        feats = np.zeros(shape, dtype=np.float32)
        start_p = 0
        if partial.exists():
            z = np.load(partial)
            if tuple(z["pert"].shape) == shape:
                feats[:] = z["pert"]
                start_p = int(z["done"])
                print(f"resume {line}/{name} from prompt {start_p}/{len(PROMPTS)}",
                      flush=True)
            else:
                print(f"partial shape {z['pert'].shape} != {shape}, restart {name}",
                      flush=True)
        for p, prompt in enumerate(PROMPTS):
            if p < start_p:
                continue
            ids = tok(prompt, return_tensors="pt").input_ids
            base = emb(ids).detach()  # [1, T, D] — batch dim required by perturb_batch
            for k, seed in enumerate(SEED_SETS):
                x = perturb_batch(base, N_MAX, seed)
                if perm_idx is not None:
                    x = x[..., perm_idx]
                feats[p, k] = pooled_layers(m, x, L4)
                if not np.isfinite(feats[p, k]).all():
                    raise SystemExit(
                        f"{line}/{name} prompt {p + 1} seed set {k}: non-finite "
                        f"features — aborting (see PLAN risk table: isolate model)")
            _save_partial(partial, feats, L4, total, p + 1)
            print(f"  {line}/{name} prompt {p + 1}/{len(PROMPTS)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        np.savez(out, pert=feats, layers4=np.array(L4), layers=np.array(L4),
                 total=total)
        partial.unlink(missing_ok=True)
        print(f"saved {out} total_layers={total} layers={L4} D={D_} "
              f"{time.time() - t0:.0f}s", flush=True)
        del m
        gc.collect()


if __name__ == "__main__":
    line = sys.argv[1]
    reg = MAIN if line == "main" else DISTILL
    names = sys.argv[2:] or list(reg)
    if line == "main":
        known = set(MAIN) | set(UNRELATED_ROUND)
        unknown = [n for n in names if n not in known]
        if unknown:
            raise SystemExit(f"unknown main-line names: {unknown}")
    import os
    torch = __import__("torch")
    # Shared box: qvh jobs already occupy ~30 cores. 8 threads is the
    # slow-collection fallback in PLAN.md; override with V4_THREADS.
    nthreads = int(os.environ.get("V4_THREADS", "8"))
    torch.set_num_threads(nthreads)
    print(f"collect threads={nthreads}", flush=True)
    collect(line, names)
