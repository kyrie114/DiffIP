"""Static diagnosis (no forward pass): tokenizer, embedding and per-layer weight drift of each
suspect vs the victim Qwen2.5-0.5B, read tensor-by-tensor from safetensors (low RAM)."""
import sys, json
from pathlib import Path
import numpy as np, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H.parent))
from common import PROMPTS, VICTIM
SUS = {"coder_base": "Qwen/Qwen2.5-Coder-0.5B", "coder_inst": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
       "r1_draft": "alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B",
       "instruct(ctrl)": "Qwen/Qwen2.5-0.5B-Instruct", "vikhr_sft(ctrl)": "Vikhrmodels/Vikhr-Qwen-2.5-0.5b-Instruct",
       "qwen2_0.5b(neg)": "Qwen/Qwen2-0.5B"}
def tensors(mid):
    d = Path(snapshot_download(mid, allow_patterns=["*.safetensors", "*.json"])); out = {}
    for f in d.glob("*.safetensors"):
        h = safe_open(str(f), "pt")
        for k in h.keys(): out[k] = (h, k)
    return out
def get(t, k): h, kk = t[k]; return h.get_tensor(kk).float()
V = tensors(VICTIM); vt = AutoTokenizer.from_pretrained(VICTIM)
vids = [vt(p)["input_ids"] for p in PROMPTS]
res = {}
for name, mid in SUS.items():
    st = tensors(mid); tk = AutoTokenizer.from_pretrained(mid); r = {"hf": mid}
    sv, ss = vt.get_vocab(), tk.get_vocab()
    r["vocab_len"] = [len(vt), len(tk)]
    r["vocab_same_id_frac"] = float(np.mean([ss.get(s) == i for s, i in sv.items()]))
    ids = [tk(p)["input_ids"] for p in PROMPTS]
    r["prompts_ids_identical"] = int(sum(a == b for a, b in zip(ids, vids)))
    used = sorted({i for x in vids for i in x})
    Ev, Es = get(V, "model.embed_tokens.weight"), get(st, "model.embed_tokens.weight")
    n = min(Ev.shape[0], Es.shape[0]); r["embed_shape"] = [list(Ev.shape), list(Es.shape)]
    if Ev.shape[1] == Es.shape[1]:
        cos = torch.nn.functional.cosine_similarity(Ev[:n], Es[:n], 1); rel = (Ev[:n] - Es[:n]).norm(dim=1) / Ev[:n].norm(dim=1).clamp_min(1e-12)
        r["embed_cos_all_mean"] = float(cos.mean()); r["embed_relL2_all_mean"] = float(rel.mean())
        r["embed_cos_prompt_tokens_mean"] = float(cos[used].mean()); r["embed_relL2_prompt_tokens_mean"] = float(rel[used].mean())
        layers = {}
        for k in V:
            if not k.startswith("model.layers.") or k not in st: continue
            li = int(k.split(".")[2]); a, b = get(V, k), get(st, k)
            if a.shape != b.shape: continue
            d = layers.setdefault(li, [0.0, 0.0]); d[0] += float((a - b).pow(2).sum()); d[1] += float(a.pow(2).sum())
        r["layer_relchange"] = [float(np.sqrt(layers[i][0] / layers[i][1])) for i in sorted(layers)]
        if "model.norm.weight" in st: r["final_norm_relchange"] = float((get(V, "model.norm.weight") - get(st, "model.norm.weight")).norm() / get(V, "model.norm.weight").norm())
    res[name] = r; print(name, {k: v for k, v in r.items() if k != "layer_relchange"}, flush=True)
    print("  layer relchange:", np.round(r.get("layer_relchange", []), 3).tolist(), flush=True)
(H / "results_static.json").write_text(json.dumps(res, indent=1))
