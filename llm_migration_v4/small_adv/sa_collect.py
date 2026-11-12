"""对每个新衍生模型采：本方法扰动特征（与 v3 collect_perturb 同协议，不存全层）、
REEF 激活（TruthfulQA 打乱后前 200 条，各层末 token，与 v3 reef_baseline 同协议）、
困惑度（TruthfulQA 前 50 条，只作健康度参考）、直接权重余弦（对受害模型）。
Usage: python sa_collect.py drop2 drop6 ...   输出 feats/ acts/ meta.json"""
import gc, json, sys, time
import numpy as np, torch
from sa_common import *  # noqa

FE = SA / "feats"; FE.mkdir(exist_ok=True)
AC = SA / "acts"; AC.mkdir(exist_ok=True)
META = SA / "meta.json"
TQA = tqa_rows()


@torch.no_grad()
def cos_table(vsd, ssd):
    vals, emb = [], None
    for k, v in vsd.items():
        s = ssd.get(k)
        if s is None or s.shape != v.shape or v.ndim < 2:
            continue
        a, b = v.flatten().double(), s.flatten().double()
        c = (a @ b / (a.norm() * b.norm())).item(); vals.append(c)
        if "embed_tokens" in k: emb = c
    n2d = sum(1 for v in vsd.values() if v.ndim >= 2)
    if not vals:
        return dict(status="not_comparable", matched=0, total=n2d)
    return dict(status="ok", mean_cos=float(np.mean(vals)), min_cos=min(vals), matched=len(vals), total=n2d, embed_cos=emb)


if __name__ == "__main__":
    meta = json.loads(META.read_text()) if META.exists() else {}
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    _vf = safe_open(hf_hub_download(c3.VICTIM, "model.safetensors"), "pt")

    class _Lazy(dict):  # 受害模型权重按需从 safetensors 读，省内存
        def items(self):
            for k in _vf.keys():
                yield "model." + k if not k.startswith(("model.", "lm_head")) else k, _vf.get_tensor(k).float()
        def values(self):
            for _, v in self.items(): yield v
    vsd = _Lazy()
    for name in sys.argv[1:]:
        t0 = time.time()
        if name in meta:
            print("skip", name, flush=True); continue
        if name == "instruct_wperm":
            m, tok, emb, P = c3.build_model(name)
        else:
            m, tok, emb = build(name); P = None
        dec = c3.decoder_layers(m); total = len(dec); L4 = c3.even_layers(total, 4)
        old = name in ("victim", "instruct", "instruct_wperm", "instruct_prune") or name in c3.NEGS
        f = FE / f"pert_{name}.npz"
        if not old and not f.exists():
            pert = []
            for p in c3.PROMPTS:
                ids = tok(p, return_tensors="pt")["input_ids"]; base = emb(ids).detach()
                sets = []
                for seed in c3.SEED_SETS:
                    x = c3.perturb_batch(base, c3.N_MAX, seed)
                    if P is not None: x = x @ P.to(x.dtype)
                    sets.append(c3.forward_pool(m, x, L4))
                pert.append(np.stack(sets))
            np.savez(f, pert=np.stack(pert).astype(np.float32), layers4=np.array(L4), total=total)
        a = AC / f"{name}.npz"
        if not old and not a.exists():
            hid = {}
            hooks = [dec[i].register_forward_hook((lambda i: lambda _m, _in, o: hid.__setitem__(i, o[0] if isinstance(o, tuple) else o))(i)) for i in range(total)]
            last = []
            with torch.no_grad():
                for t in TQA[:200]:
                    ids = tok(t, return_tensors="pt")["input_ids"]; e = emb(ids).detach()
                    if P is not None: e = e @ P.to(e.dtype)
                    m(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], dtype=torch.long), use_cache=False)
                    last.append(torch.stack([hid[i][0, -1] for i in range(total)]).float())
            for h in hooks: h.remove()
            np.savez(a, tqa200_last=torch.stack(last, 1).numpy(), total=total)
        ent = dict(layers=total, layers4=L4, hidden=int(m.get_input_embeddings().weight.shape[1]),
                   vocab=int(m.get_input_embeddings().weight.shape[0]),
                   ppl_tqa50=None if P is not None else ppl(m, tok, TQA[:50]),
                   weight_cos=cos_table(vsd, m.state_dict()))
        meta[name] = ent; META.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        print(name, {k: v for k, v in ent.items() if k != "weight_cos"}, ent["weight_cos"], f"{time.time()-t0:.0f}s", flush=True)
        del m; gc.collect()
