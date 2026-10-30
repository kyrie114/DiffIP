"""Distillation line, changed signal (CPU, no training). Idea in the spirit of output-distribution
/ teacher-bias fingerprints for distillation: instead of hidden-state geometry, compare what the
suspect *outputs*. Teacher = Qwen2.5-0.5B-Instruct. Suspects = existing ckpts from
/workspace/papers/exp_distill_v2 (student_distill = SmolLM2-135M SFT on teacher answers;
student_control = same base/budget on original Alpaca answers), SmolLM2-135M base, and negatives.
Data: the 120 held-out instructions (not used in training) from exp_distill_v2/data/teacher_data.json,
prompt template '### Instruction:\n...\n\n### Response:\n' (same as training).
S1 (perturbation cloud, next-token distribution): at the response-start position, each model's own
input embeddings get the v3 perturbation (seed 123, scale 2e-3, N=16 incl. clean). Distributions are
mapped to a shared space = decoded token strings present in BOTH vocabularies (stripped of BPE space
markers is NOT done: exact decoded string match), renormalised; per-prompt score = 1 - JS(teacher, suspect)
averaged over the 16 cloud members (paired by index; noise is not shared across models).
S2 (teacher-bias, tokenizer-agnostic): -(per-byte NLL) of the teacher's own held-out answer under the
suspect (teacher forcing, first 256 suspect tokens).
Detection: positive = student_distill; metrics per prompt vs each other model; AUC vs pooled negatives
(student_control, base, neg_*), gap = mean(distill) - max mean(other)."""
import json, math, sys, gc
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
torch.set_num_threads(8); torch.set_grad_enabled(False)
H = Path(__file__).resolve().parent; D = Path("/workspace/papers/exp_distill_v2")
HO = json.load(open(D / "data/teacher_data.json"))["heldout"]
def fmt(ins): return f"### Instruction:\n{ins}\n\n### Response:\n"
SUS = {"student_distill": str(D / "ckpt/student_distill"), "student_control": str(D / "ckpt/student_control"),
       "base_smollm2_135m": "HuggingFaceTB/SmolLM2-135M", "neg_smollm2_360m": "HuggingFaceTB/SmolLM2-360M",
       "neg_gpt2": "openai-community/gpt2", "neg_pythia_160m": "EleutherAI/pythia-160m", "neg_qwen2_0.5b": "Qwen/Qwen2-0.5B"}
TEACHER = "Qwen/Qwen2.5-0.5B-Instruct"; NP = 16
OUT = H / "feats_distill_signal"; OUT.mkdir(exist_ok=True)

def collect(key, path):
    f = OUT / f"{key}.npz"
    if f.exists(): return
    tok = AutoTokenizer.from_pretrained(path); m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    E = m.get_input_embeddings(); vocab = [tok.decode([i]) for i in range(len(tok))]
    probs, nll = [], []
    for r in HO:
        ids = tok(fmt(r["instruction"]), return_tensors="pt")["input_ids"]; b = E(ids)
        g = torch.Generator().manual_seed(123); xs = [b] + [b + 2e-3 * torch.randn(b.shape, generator=g) for _ in range(NP - 1)]
        lg = m(inputs_embeds=torch.cat(xs), use_cache=False).logits[:, -1].float()
        probs.append(torch.softmax(lg, -1)[:, :len(vocab)].numpy().astype(np.float32))
        p = tok(fmt(r["instruction"]))["input_ids"]; a = tok(r["teacher"] + (tok.eos_token or ""))["input_ids"][:256]
        t = torch.tensor([p + a]); lp = torch.log_softmax(m(t, use_cache=False).logits[0, len(p) - 1:-1].float(), -1)
        s = -lp[torch.arange(len(a)), torch.tensor(a)].sum().item(); nbytes = len(tok.decode(a).encode()) or 1
        nll.append(s / nbytes)
    np.savez_compressed(f, probs=np.stack(probs), nll=np.array(nll), vocab=np.array(vocab, dtype=object))
    print("collected", key, flush=True); del m; gc.collect()

def shared(pa, va, pb, vb):
    ia = {}; 
    for i, s in enumerate(va): ia.setdefault(s, i)
    ib = {}
    for i, s in enumerate(vb): ib.setdefault(s, i)
    keys = [s for s in ia if s in ib]; A = pa[..., [ia[s] for s in keys]]; B = pb[..., [ib[s] for s in keys]]
    A = A / A.sum(-1, keepdims=True).clip(1e-12); B = B / B.sum(-1, keepdims=True).clip(1e-12)
    M = 0.5 * (A + B); kl = lambda P, Q: (P * (np.log(P.clip(1e-12)) - np.log(Q.clip(1e-12)))).sum(-1)
    js = 0.5 * kl(A, M) + 0.5 * kl(B, M)  # nats, in [0, ln2]
    return 1 - js / math.log(2), len(keys), (A.argmax(-1) == B.argmax(-1))

if __name__ == "__main__":
    collect("teacher", TEACHER)
    for k, p in SUS.items(): collect(k, p)
    T = np.load(OUT / "teacher.npz", allow_pickle=True); res = {"models": SUS, "teacher": TEACHER, "n_prompts": len(HO), "cloud": NP, "rows": {}}
    S1, S2 = {}, {}
    for k in SUS:
        z = np.load(OUT / f"{k}.npz", allow_pickle=True)
        sim, nk, agree = shared(T["probs"], list(T["vocab"]), z["probs"], list(z["vocab"]))
        S1[k] = sim.mean(1); S2[k] = -z["nll"]
        res["rows"][k] = dict(shared_vocab=nk, s1_mean=float(S1[k].mean()), top1_agree=float(agree.mean()), teacher_nll_per_byte=float(z["nll"].mean()))
    auc = lambda s, n: float(((s[:, None] > n[None]) + 0.5 * (s[:, None] == n[None])).mean())
    for nm, S in (("S1_cloud_JS", S1), ("S2_teacher_nll", S2)):
        pos = S["student_distill"]; others = {k: v for k, v in S.items() if k != "student_distill"}
        res[nm] = dict(gap=float(pos.mean() - max(v.mean() for v in others.values())), auc_vs_pooled=auc(pos, np.concatenate(list(others.values()))),
            auc_vs_control=auc(pos, S["student_control"]), win_vs_control=int((pos > S["student_control"]).sum()),
            win_vs_all=int((pos > np.max(np.stack(list(others.values())), 0)).sum()))
    (H / "results_distill_signal.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "models"}, indent=1))
