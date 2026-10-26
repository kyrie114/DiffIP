"""Mode (b): stochastic-decoding randomness. Seed -> uniforms u[i,t] (shape [N_MAX, T_GEN]),
shared by suspect and victim. Sample i draws token t by inverse-CDF on the model's own
temperature-1 next-token distribution with u[i,t] (the same seed => the same random
stream, the strongest pairing available across different vocabularies). Representation
of sample i = attention-mask mean pool of decoder-layer outputs over prompt+continuation,
4 even layers. Budget-limited: 8 prompts, N_MAX=64, 3 seed sets."""
import sys, time
from common import *
T_GEN = 16; NMAX_DEC = 64
DEC_PROMPTS = [0, 1, 7, 8, 13, 19, 25, 30]  # en,en,zh,zh,code,math,qa,qa(zh)
OUT = H / "feats"
@torch.no_grad()
def generate(m, ids, u):
    seq = ids.repeat(u.shape[0], 1)
    for t in range(u.shape[1]):
        try:
            logits = m(input_ids=seq, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        except TypeError:
            logits = m(input_ids=seq, use_cache=False).logits[:, -1].float()
        cdf = torch.softmax(logits, -1).cumsum(-1)
        nxt = torch.searchsorted(cdf, (u[:, t:t+1] * cdf[:, -1:]).contiguous()).clamp_max(cdf.shape[1]-1)
        seq = torch.cat([seq, nxt], 1)
    return seq
names = sys.argv[1:] or ["victim", "instruct", "instruct_prune", *NEGS]
for name in names:
    f = OUT / f"dec_{name}.npz"
    if f.exists(): print("skip", name); continue
    t0 = time.time(); m, tok, emb, _ = build_model(name)
    total = len(decoder_layers(m)); L4 = even_layers(total, 4)
    allp, texts = [], []
    for pi in DEC_PROMPTS:
        ck = OUT / f"dec_{name}_p{pi}.npz"
        if ck.exists():
            z = np.load(ck); allp.append(z["r"]); texts.append(str(z["t"])); continue
        ids = tok(PROMPTS[pi], return_tensors="pt")["input_ids"]; sets = []
        for seed in SEED_SETS:
            u = torch.rand((NMAX_DEC, T_GEN), generator=torch.Generator().manual_seed(seed))
            seq = generate(m, ids, u)
            sets.append(forward_pool(m, emb(seq).detach(), L4))
            if seed == SEED_SETS[0]: texts.append(tok.decode(seq[0, ids.shape[1]:]))
        allp.append(np.stack(sets)); np.savez(ck, r=allp[-1], t=texts[-1]); print(name, "prompt", pi, f"{time.time()-t0:.0f}s", flush=True)
    np.savez(f, pert=np.stack(allp), layers4=np.array(L4), total=total, sample_text=np.array(texts))
    print(name, f"{time.time()-t0:.0f}s", texts[:2], flush=True); del m; gc.collect()
