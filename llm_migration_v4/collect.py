"""Collect fingerprint features for a line.

Usage:
    python collect.py main [names...]      # default: all MAIN models
    python collect.py distill <names...>   # names must exist (students trained)

Output: feats/{line}_{name}.npz with pert [32 prompts, 3 seed sets, 4 layers, 128, D],
layers4, total. Reproducible per-prompt fingerprints (v3 protocol scheme a).
"""
import gc
import sys
import time

import numpy as np

from common import (H, MAIN, DISTILL, N_LAYERS, N_MAX, PROMPTS, SEED_SETS,
                    build_main, even_layers, load, pooled_layers, perturb_batch)

OUT = H / "feats"
OUT.mkdir(exist_ok=True)


def collect(line, names):
    for name in names:
        out = OUT / f"{line}_{name}.npz"
        if out.exists():
            print("skip", name, flush=True)
            continue
        t0 = time.time()
        if line == "main":
            m, tok, emb, perm_idx = build_main(name)
        else:
            from common import DISTILL as D
            m, tok = load(D[name])
            E = m.get_input_embeddings()
            emb, perm_idx = (lambda ids: E(ids)), None
        total = len(m.model.layers)
        L4 = even_layers(total, N_LAYERS)
        D_ = m.config.hidden_size
        feats = np.zeros((len(PROMPTS), len(SEED_SETS), N_LAYERS, N_MAX, D_),
                         dtype=np.float32)
        for p, prompt in enumerate(PROMPTS):
            ids = tok(prompt, return_tensors="pt").input_ids
            base = emb(ids).detach()  # [1, T, D] — batch dim required by perturb_batch
            for k, seed in enumerate(SEED_SETS):
                x = perturb_batch(base, N_MAX, seed)
                if perm_idx is not None:
                    x = x[..., perm_idx]
                feats[p, k] = pooled_layers(m, x, L4)
            print(f"  {line}/{name} prompt {p + 1}/{len(PROMPTS)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        np.savez(out, pert=feats, layers4=np.array(L4), total=total)
        print(f"saved {out} total_layers={total} L4={L4} D={D_} "
              f"{time.time() - t0:.0f}s", flush=True)
        del m
        gc.collect()


if __name__ == "__main__":
    line = sys.argv[1]
    reg = MAIN if line == "main" else DISTILL
    names = sys.argv[2:] or list(reg)
    torch = __import__("torch")
    torch.set_num_threads(16)
    import torch  # noqa
    collect(line, names)
