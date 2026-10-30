"""Collect perturbation features for public derivatives / hard negatives; protocol identical
to ../collect_perturb.py and ../fingerinv_attacks/collect.py (32 prompts, 3 seed sets, N=128,
scale 2e-3, 4 evenly spaced decoder layers, mean pooling, fp32). Writes feats/pert_<key>.npz,
logs failures to failures.json and keeps going."""
import sys, time, json, traceback, shutil
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from common import *
sys.path.insert(0, str(Path(__file__).resolve().parent)); from lit_models import MODELS
OUT = Path(__file__).resolve().parent / "feats"; OUT.mkdir(exist_ok=True)
FAIL = Path(__file__).resolve().parent / "failures.json"
fails = json.loads(FAIL.read_text()) if FAIL.exists() else {}
for key in (sys.argv[1:] or MODELS):
    f = OUT / f"pert_{key}.npz"
    if f.exists(): print("skip", key); continue
    t0 = time.time()
    try:
        m, tok = load(MODELS[key][0]); E = m.get_input_embeddings()
        total = len(decoder_layers(m)); L4 = even_layers(total, 4); pert = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt")["input_ids"]
            base = E(ids).detach()
            pert.append(np.stack([forward_pool(m, perturb_batch(base, N_MAX, s), L4, chunk=32) for s in SEED_SETS]))
        np.savez(f, pert=np.stack(pert).astype(np.float32), layers4=np.array(L4), total=total)
        print(key, "L4", L4, f"{time.time()-t0:.0f}s", flush=True); fails.pop(key, None)
        del m
    except Exception as e:
        fails[key] = repr(e)[:400]; print("FAIL", key, e, flush=True); traceback.print_exc()
    FAIL.write_text(json.dumps(fails, indent=1)); gc.collect()
