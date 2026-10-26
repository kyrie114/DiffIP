"""Mode (a): paired-seed input-embedding perturbation. For every model, every prompt
(own fingerprint per prompt), 3 seed sets x N_MAX=128 samples at 4 even layers.
N<128 uses the first N samples of a set, which is bit-identical to running the
collector with n_samples=N and the same seed. Also stores all-layer reps for the
first 20 samples of seed set 0 (paper N=20) for the DP-on experiment."""
import sys, time
from common import *
OUT = H / "feats"; OUT.mkdir(exist_ok=True)
names = sys.argv[1:] or ["victim", "instruct", "instruct_wperm", "instruct_prune", *NEGS]
for name in names:
    f = OUT / f"pert_{name}.npz"
    if f.exists(): print("skip", name); continue
    t0 = time.time()
    m, tok, emb, P = build_model(name)
    total = len(decoder_layers(m)); L4 = even_layers(total, 4); allL = list(range(total))
    pert, dpall = [], []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt")["input_ids"]
        base = emb(ids).detach()
        sets = []
        for k, seed in enumerate(SEED_SETS):
            x = perturb_batch(base, N_MAX, seed)
            if P is not None: x = x @ P.to(x.dtype)
            if k == 0:
                r = forward_pool(m, x, allL)
                dpall.append(r[:, :20].astype(np.float32)); sets.append(r[L4])
            else:
                sets.append(forward_pool(m, x, L4))
        pert.append(np.stack(sets))
    np.savez(f, pert=np.stack(pert), dp_all=np.stack(dpall), layers4=np.array(L4), total=total)
    print(name, "total layers", total, "L4", L4, "D", pert[0].shape[-1], f"{time.time()-t0:.0f}s", flush=True)
    del m; gc.collect()
