"""步骤2：SmolLM2-135M 学生（与 Qwen 异架构/异词表：d=576, 30 层, vocab 49152）。
mode=distill：在教师回答上做序列级蒸馏（Kim&Rush 2016；词表不同无法逐 token 对齐 logits，故用教师贪心输出的 CE）。
mode=control：同样的指令、同样步数/超参，但回答换成 Alpaca 原始回答（非教师）——独立负样本。只对回答部分计损失。"""
import json, sys, time, math, random, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(8); torch.manual_seed(0); random.seed(0)
MODE, EPOCHS = sys.argv[1], int(sys.argv[2]); KEY = "teacher" if MODE == "distill" else "alpaca"
S = "HuggingFaceTB/SmolLM2-135M"; tok = AutoTokenizer.from_pretrained(S); m = AutoModelForCausalLM.from_pretrained(S, dtype=torch.float32)
D = json.load(open("data/teacher_data.json"))["train"]
def fmt(ins): return f"### Instruction:\n{ins}\n\n### Response:\n"
ex = []
for r in D:
    p = tok(fmt(r["instruction"]))["input_ids"]; a = tok(r[KEY] + tok.eos_token)["input_ids"]
    ex.append(((p + a)[:320], len(p)))
BS, LR = 8, 1e-4; steps = EPOCHS * math.ceil(len(ex) / BS)
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0)
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * max(0.0, 1 - s / steps))
m.train(); st = 0; t0 = time.time(); log = []
for ep in range(EPOCHS):
    idx = np.random.RandomState(ep).permutation(len(ex)); tot = 0; nb = 0
    for i in range(0, len(idx), BS):
        b = [ex[j] for j in idx[i:i+BS]]; L = max(len(s) for s, _ in b)
        ii = torch.full((len(b), L), tok.eos_token_id); am = torch.zeros_like(ii); lab = torch.full_like(ii, -100)
        for k, (s, pl) in enumerate(b):
            ii[k, :len(s)] = torch.tensor(s); am[k, :len(s)] = 1; lab[k, pl:len(s)] = torch.tensor(s[pl:])
        loss = m(input_ids=ii, attention_mask=am, labels=lab).loss; loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); st += 1; tot += loss.item(); nb += 1
        if st % 25 == 0: print(f"ep{ep} step{st}/{steps} loss{tot/nb:.4f} {time.time()-t0:.0f}s", flush=True)
    log.append(dict(epoch=ep, mean_loss=tot / nb)); print("EPOCH", ep, tot / nb, flush=True)
m.save_pretrained(f"ckpt/student_{MODE}"); tok.save_pretrained(f"ckpt/student_{MODE}")
json.dump(dict(mode=MODE, epochs=EPOCHS, steps=steps, bs=BS, lr=LR, n_train=len(ex), log=log, sec=time.time() - t0), open(f"ckpt/train_{MODE}.json", "w"), indent=1)
