"""REEF activations: for each model, feed the SAME inputs (200 TruthfulQA statements,
random.Random(2024) sample, statement = Question + ' ' + Best Answer; plus the 32 v3 prompts),
one at a time without padding (as official generate_activations.get_acts), hook every decoder
layer, keep last-token activation (official REEF, token_pos=-1) and token-mean pool (v3 style).
Models built exactly as v3 common.build_model (victim / instruct / instruct_wperm / instruct_prune / 5 negs)."""
import sys, csv, random, time
from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from common import *
OUT = H / "reef_baseline" / "acts"; OUT.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(H / "reef_baseline" / "TruthfulQA.csv")))
random.Random(2024).shuffle(rows)
TQA = [r["Question"].strip() + " " + r["Best Answer"].strip() for r in rows[:200]]
INPUTS = {"tqa200": TQA, "prompts32": PROMPTS}
names = sys.argv[1:] or ["victim", "instruct", "instruct_wperm", "instruct_prune", *NEGS]
for name in names:
    f = OUT / f"{name}.npz"
    if f.exists(): print("skip", name); continue
    t0 = time.time(); m, tok, emb, P = build_model(name)
    allL = list(range(len(decoder_layers(m)))); dec = decoder_layers(m); hid = {}
    hooks = [dec[i].register_forward_hook((lambda i: lambda _m, _in, o: hid.__setitem__(i, o[0] if isinstance(o, tuple) else o))(i)) for i in allL]
    res = {}
    with torch.no_grad():
        for key, texts in INPUTS.items():
            last, mean = [], []
            for t in texts:
                ids = tok(t, return_tensors="pt")["input_ids"]
                e = emb(ids).detach()
                if P is not None: e = e @ P.to(e.dtype)
                m(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], dtype=torch.long), use_cache=False)
                last.append(torch.stack([hid[i][0, -1] for i in allL]).float())
                mean.append(torch.stack([hid[i][0].mean(0) for i in allL]).float())
            res[f"{key}_last"] = torch.stack(last, 1).numpy(); res[f"{key}_mean"] = torch.stack(mean, 1).numpy()
    for h in hooks: h.remove()
    np.savez(f, **res, total=len(allL))
    print(name, len(allL), f"{time.time()-t0:.0f}s", flush=True); del m; gc.collect()
