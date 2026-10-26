"""步骤3：蒸馏强度。120 条留出指令（训练未见）。
(a) 教师强制：学生在教师回答上的每字节 NLL、逐 token top-1 命中率（= 与教师贪心选择的一致率，在学生分词下）；
(b) 对照：同样指标在 Alpaca 原始回答上；(c) 学生贪心生成 vs 教师回答的 ROUGE-L F1（词级 LCS），前 60 条。"""
import json, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(8); torch.set_grad_enabled(False)
H = json.load(open("data/teacher_data.json"))["heldout"]
def fmt(ins): return f"### Instruction:\n{ins}\n\n### Response:\n"
def lcs(a, b):
    d = [0] * (len(b) + 1)
    for x in a:
        p = 0
        for j, y in enumerate(b):
            t = d[j + 1]; d[j + 1] = p + 1 if x == y else max(d[j + 1], d[j]); p = t
    return d[-1]
def rouge(h, r):
    h, r = h.lower().split(), r.lower().split(); l = lcs(h, r)
    return 0.0 if l == 0 else 2 * l / (len(h) + len(r))
def run(name, path):
    tok = AutoTokenizer.from_pretrained(path); m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval(); R = {"model": name}
    for key in ("teacher", "alpaca"):
        nll = 0; nb = 0; hit = 0; nt = 0
        for r in H:
            p = tok(fmt(r["instruction"]))["input_ids"]; a = tok(r[key] + tok.eos_token)["input_ids"]; ids = torch.tensor([p + a])
            lp = torch.log_softmax(m(ids).logits[0, len(p) - 1:-1].float(), -1); tgt = ids[0, len(p):]
            nll += -lp[torch.arange(len(tgt)), tgt].sum().item(); nb += len((r[key]).encode()) + 1
            hit += (lp.argmax(-1) == tgt).sum().item(); nt += len(tgt)
        R[f"{key}_nll_per_byte"] = nll / nb; R[f"{key}_top1"] = hit / nt
    rs, rs_a = [], []; gens = []
    for r in H[:60]:
        enc = tok(fmt(r["instruction"]), return_tensors="pt")
        g = tok.decode(m.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        g = g.split("### Instruction")[0].strip(); gens.append(g)
        rs.append(rouge(g, r["teacher"])); rs_a.append(rouge(g, r["alpaca"]))
    R["rougeL_vs_teacher"] = float(np.mean(rs)); R["rougeL_vs_alpaca"] = float(np.mean(rs_a)); R["gens"] = gens[:10]
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in R.items() if k != "gens"}, flush=True); return R
M = {"smollm_base": "HuggingFaceTB/SmolLM2-135M", "student_distill": "ckpt/student_distill", "student_control": "ckpt/student_control"}
out = [run(n, p) for n, p in M.items()]
out.append(dict(model="alpaca_ref_vs_teacher_rougeL", value=float(np.mean([rouge(r["alpaca"], r["teacher"]) for r in H[:60]]))))
json.dump(out, open("strength.json", "w"), indent=1, ensure_ascii=False)
