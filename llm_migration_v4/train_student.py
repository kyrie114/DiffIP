"""Train the 1B students (Llama-3.2-1B base) on the v4 teacher data.

mode=distill : sequence-level distillation on Llama-3.2-3B teacher answers
mode=control : same instructions / hyperparams / step budget, but trained on
               the original Alpaca answers (independent negative control)

Chat-template format (Llama-3.2). Loss on the answer span only. CPU fp32.
Usage: python train_student.py distill 3 | python train_student.py control 3
"""
import json
import sys
import time
import math
import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from common import H, MODELS, load_tok

torch.set_num_threads(12)
torch.manual_seed(0)
random.seed(0)
MODE, EPOCHS = sys.argv[1], int(sys.argv[2])
KEY = "teacher" if MODE == "distill" else "alpaca"
S = str(MODELS / "Llama-3.2-1B")
tok = load_tok(S)
m = AutoModelForCausalLM.from_pretrained(S, dtype=torch.float32)
D = json.load(open(H / "data" / "teacher_data.json"))["train"]


# Llama-3.2-1B local copy is a BASE model (no chat tuning) — use v1's plain
# Alpaca completion format for both students (identical to llm_migration v1).
def fmt(ins):
    return f"### Instruction:\n{ins}\n\n### Response:\n"


ex = []
for r in D:
    p = tok(fmt(r["instruction"]))["input_ids"]
    a = tok(r[KEY] + tok.eos_token)["input_ids"]
    ex.append(((p + a)[:320], len(p)))

BS, LR = 8, 1e-4
steps = EPOCHS * math.ceil(len(ex) / BS)
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0)
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * max(0.0, 1 - s / steps))
m.train()
st = 0
t0 = time.time()
for ep in range(EPOCHS):
    idx = np.random.RandomState(ep).permutation(len(ex))
    tot = 0
    nb = 0
    for i in range(0, len(idx), BS):
        b = [ex[j] for j in idx[i:i + BS]]
        L = max(len(s) for s, _ in b)
        ii = torch.full((len(b), L), tok.eos_token_id)
        am = torch.zeros_like(ii)
        lab = torch.full_like(ii, -100)
        for k, (s, pl) in enumerate(b):
            ii[k, :len(s)] = torch.tensor(s)
            am[k, :len(s)] = 1
            lab[k, pl:len(s)] = torch.tensor(s[pl:])
        loss = m(input_ids=ii, attention_mask=am, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        opt.zero_grad()
        st += 1
        tot += loss.item()
        nb += 1
        if st % 25 == 0:
            print(f"ep{ep} step{st}/{steps} loss{tot / nb:.4f} {time.time() - t0:.0f}s",
                  flush=True)
    print("EPOCH", ep, tot / nb, flush=True)

( H / "ckpt" ).mkdir(exist_ok=True)
m.save_pretrained(H / "ckpt" / f"student_{MODE}")
tok.save_pretrained(H / "ckpt" / f"student_{MODE}")
json.dump(dict(mode=MODE, epochs=EPOCHS, steps=steps, bs=BS, lr=LR,
               n_train=len(ex), sec=time.time() - t0),
          open(H / "ckpt" / f"train_{MODE}.json", "w"), indent=1)
print("saved", H / "ckpt" / f"student_{MODE}", flush=True)
