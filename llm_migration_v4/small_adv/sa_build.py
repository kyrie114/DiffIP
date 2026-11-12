"""在本机构造宽度剪枝与换词表衍生模型，存 ckpt/{name}/（model.pt + tok/）。
Usage: python sa_build.py width75 width50 vocab_ext vocab_swap"""
import json, sys, time
import torch
from sa_common import *  # noqa

STEPS = 300
log_f = open(SA / "build.log", "a")
def log(s):
    print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

en, zh = corpus()
log(f"corpus: en {len(en)} 段, zh {len(zh)} 段")
for name in sys.argv[1:]:
    t0 = time.time(); d = CKPT / name
    if d.exists(): log(f"skip {name}"); continue
    m, tok = c3.load(c3.INSTRUCT)
    info = {}
    if name.startswith("width"):
        P = int(name[5:].rstrip("h")); D = m.config.hidden_size; k = round(D * P / 100)
        imp = channel_importance(m, tok, en[:64])
        keep = torch.argsort(imp, descending=True)[:k]
        width_prune(m, keep); info = dict(kept=k, of=D, calib="TruthfulQA 第 200 条后 64 条")
        if name.endswith("h"):
            train_embeddings(m, tok, en + zh, STEPS, lr=5e-5, log=log, which="body")
    elif name == "vocab_ext":
        merges = merge_tokens(tok, en + zh, 2000)
        E = m.model.embed_tokens.weight.data
        tok.add_tokens([s for s, _, _ in merges])
        V_new = len(tok)
        W = torch.zeros(V_new, E.shape[1]); W[:E.shape[0]] = E
        for s, a, b in merges:
            W[tok.convert_tokens_to_ids(s)] = (E[a] + E[b]) / 2
        set_vocab(m, W)
        orig = AutoTokenizer.from_pretrained(c3.INSTRUCT)
        changed = sum(tok(p).input_ids != orig(p).input_ids for p in c3.PROMPTS)
        info = dict(old_vocab=int(E.shape[0]), new_vocab=V_new, added=len(merges), prompts_retokenized=int(changed))
        train_embeddings(m, tok, en + zh, STEPS, log=log)
    elif name == "vocab_swap":
        qtok = tok; stok = AutoTokenizer.from_pretrained(SMOL)
        E = m.model.embed_tokens.weight.data; qv = qtok.get_vocab(); sv = stok.get_vocab()
        W = torch.zeros(len(stok), E.shape[1]); exact = 0; mean_init = 0; fallback = 0
        for s, i in sv.items():
            if s in qv and qv[s] < E.shape[0]:
                W[i] = E[qv[s]]; exact += 1; continue
            txt = stok.convert_tokens_to_string([s])
            ids = qtok(txt, add_special_tokens=False).input_ids
            if ids: W[i] = E[ids].mean(0); mean_init += 1
            else: W[i] = E.mean(0); fallback += 1
        set_vocab(m, W); tok = stok
        info = dict(old_vocab=int(E.shape[0]), new_vocab=len(stok), same_string_copied=exact,
                    subword_mean_init=mean_init, fallback=fallback)
        train_embeddings(m, tok, en + zh, STEPS, log=log)
    else:
        raise SystemExit(name)
    d.mkdir(parents=True); tok.save_pretrained(d / "tok")
    if name.startswith("width"):
        r = dict(keep=keep.clone())
        if name.endswith("h"):
            r["body"] = {k: v.clone() for k, v in m.state_dict().items() if "embed_tokens" not in k and "lm_head" not in k}
        torch.save(r, d / "recipe.pt")
    else:
        torch.save(dict(embed=m.model.embed_tokens.weight.data.clone()), d / "recipe.pt")
    info["train_steps"] = STEPS if (name.startswith("vocab") or name.endswith("h")) else 0
    (d / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    log(f"{name} {info} {time.time()-t0:.0f}s")
