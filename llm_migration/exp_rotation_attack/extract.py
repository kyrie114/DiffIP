"""真实实验：对 Qwen2.5-0.5B-Instruct 做保函数的隐藏空间变换（置换 / 折叠 RMSNorm 后随机正交旋转），
验证函数等价，并抽取逐层 last-token hidden。CPU fp32。"""
import os, json, gc, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(0); torch.set_num_threads(8)
OUT = "/workspace/papers/exp_rotation_attack"; os.makedirs(f"{OUT}/feats", exist_ok=True)
_P = json.load(open("/workspace/20261019_DiffIP迁移LLM_线性回拉/prompts.json")); PROMPTS = _P["calibration"] + _P["test"]   # 72 条，与之前实验相同
VERIFY = PROMPTS[:5] + PROMPTS[36:41]                                                            # 10 条验证提示
SRC, DER, IND = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2-0.5B"

def load(mid):
    return AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()

@torch.no_grad()
def extract(model, tok):
    rows = []
    for i in range(0, len(PROMPTS), 8):
        enc = tok(PROMPTS[i:i+8], return_tensors="pt", padding=True)
        H = torch.stack(model(**enc, output_hidden_states=True).hidden_states, 1)
        lens = enc["attention_mask"].sum(1) - 1
        rows += [H[b, :, lens[b], :].numpy() for b in range(H.shape[0])]
    return np.stack(rows).astype(np.float32)

@torch.no_grad()
def fingerprint_outputs(model, tok):
    logits, gens = [], []
    for p in VERIFY:
        enc = tok(p, return_tensors="pt")
        logits.append(model(**enc).logits[0].clone())
        g = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        gens.append(tok.decode(g[0, enc["input_ids"].shape[1]:]))
    return logits, gens

@torch.no_grad()
def transform(model, Q, fold):
    """h' = h Q。读残差的线性层 W(out×in) -> W Q；写残差的 o/down W -> Q^T W；嵌入 E -> E Q。
    fold=True：先把每个 RMSNorm 的逐通道权重并入其后的线性层，norm 权重置 1（SliceGPT/QuaRot 做法），再做任意正交旋转。
    fold=False：只允许置换（及符号翻转需折叠，这里不用），RMSNorm 权重同步置换。"""
    m = model.model
    if model.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr():          # Qwen2.5-0.5B 共享嵌入：先解绑
        model.lm_head.weight = torch.nn.Parameter(m.embed_tokens.weight.detach().clone())
        model.config.tie_word_embeddings = False
    perm = None if fold else Q.argmax(0)       # Q[perm[j], j] = 1 → (hQ)_j = h_perm[j]
    def norm_and_reads(norm, linears):
        w = norm.weight.data
        for L in linears:
            W = L.weight.data * w[None, :] if fold else L.weight.data
            L.weight.data = W @ Q
        norm.weight.data = torch.ones_like(w) if fold else w[perm]
    m.embed_tokens.weight.data = m.embed_tokens.weight.data @ Q
    for layer in m.layers:
        a, f = layer.self_attn, layer.mlp
        norm_and_reads(layer.input_layernorm, [a.q_proj, a.k_proj, a.v_proj])
        norm_and_reads(layer.post_attention_layernorm, [f.gate_proj, f.up_proj])
        for L in (a.o_proj, f.down_proj):
            L.weight.data = Q.T @ L.weight.data
            if L.bias is not None: L.bias.data = L.bias.data @ Q
    norm_and_reads(m.norm, [model.lm_head])
    return model

def main():
    t0 = time.time(); log = {}
    tok = AutoTokenizer.from_pretrained(SRC)
    for name, mid in (("source", SRC), ("independent", IND)):
        mdl = load(mid); np.save(f"{OUT}/feats/{name}.npy", extract(mdl, AutoTokenizer.from_pretrained(mid))); del mdl; gc.collect(); print(name, "done", flush=True)
    dtok = AutoTokenizer.from_pretrained(DER)
    D = 896; g = torch.Generator().manual_seed(0)
    Qp = torch.eye(D)[:, torch.randperm(D, generator=g)]
    Qr, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64)); Qr = Qr.float()
    ref_logits = ref_gens = None
    for name, Q, fold in (("derived", None, None), ("derived_perm", Qp, False), ("derived_rot", Qr, True), ("derived_foldonly", torch.eye(D), True)):
        mdl = load(DER)
        if Q is not None: transform(mdl, Q, fold)
        lg, gens = fingerprint_outputs(mdl, dtok)
        if ref_logits is None: ref_logits, ref_gens = lg, gens
        else:
            log[name] = dict(max_abs_logit_diff=max(float((a - b).abs().max()) for a, b in zip(lg, ref_logits)),
                             mean_abs_logit_diff=float(np.mean([float((a - b).abs().mean()) for a, b in zip(lg, ref_logits)])),
                             ref_logit_abs_max=max(float(a.abs().max()) for a in ref_logits),
                             argmax_agree=float(np.mean([float((a.argmax(-1) == b.argmax(-1)).float().mean()) for a, b in zip(lg, ref_logits)])),
                             greedy_identical=f"{sum(x == y for x, y in zip(gens, ref_gens))}/{len(gens)}")
            print(name, log[name], flush=True)
        if Q is not None and not fold:     # 置换不改范数：hidden 应恰为原 hidden 的列置换
            pass
        np.save(f"{OUT}/feats/{name}.npy", extract(mdl, dtok)); del mdl; gc.collect()
    log["greedy_examples"] = [dict(prompt=p, derived=g) for p, g in zip(VERIFY[:3], ref_gens[:3])]
    log["orthogonality_err_rot"] = float((Qr.T @ Qr - torch.eye(D)).abs().max()); log["runtime_sec"] = time.time() - t0
    json.dump(log, open(f"{OUT}/verify.json", "w"), indent=1, ensure_ascii=False)
main()
