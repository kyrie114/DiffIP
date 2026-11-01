"""Teacher (Llama-3.2-3B) generates greedy answers for the v1 Alpaca split
(1600 train + 120 heldout instructions; original Alpaca answers kept as the
control student's training text). CPU bf16.

Output: data/teacher_data.json  {train:[{instruction, teacher, alpaca}], heldout:[...]}

Progress is appended to data/teacher_data.generating.json and only moved onto
teacher_data.json when both splits are complete, so a concurrent control
training run keeps the intact Alpaca copy, and a killed run resumes by row.
"""
import json
import os
import time
import random

import torch
from transformers import AutoModelForCausalLM

from common import H, MODELS, ROOT, load_tok

NTHREADS = int(os.environ.get("V4_THREADS", "6"))
torch.set_num_threads(NTHREADS)
random.seed(0)
MAXNEW = 64
BS = 32
TEACHER = str(MODELS / "Llama-3.2-3B")
SRC = ROOT / "llm_migration" / "exp_distill_v2" / "data" / "teacher_data.json"
OUT = H / "data" / "teacher_data.json"
PARTIAL = H / "data" / "teacher_data.generating.json"

# Llama-3.2-1B/3B local copies are BASE models (no chat tuning) — the chat
# template is meaningless to them. Use v1's plain Alpaca completion format.
FMT = "### Instruction:\n{}\n\n### Response:\n"


def main():
    print(f"gen_teacher threads={NTHREADS}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    tok = load_tok(TEACHER)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).eval()
    src = json.load(open(SRC))
    if PARTIAL.exists():
        out = json.load(open(PARTIAL))
        print(f"resume train {len(out['train'])}/{len(src['train'])} "
              f"heldout {len(out['heldout'])}/{len(src['heldout'])}", flush=True)
    else:
        out = {"train": [], "heldout": []}
    t0 = time.time()
    for split in ("train", "heldout"):
        rows = src[split]
        done = len(out[split])
        for i in range(done, len(rows), BS):
            b = rows[i:i + BS]
            ps = [FMT.format(r["instruction"]) for r in b]
            enc = tok(ps, return_tensors="pt", padding=True)
            with torch.no_grad():
                g = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                               pad_token_id=tok.pad_token_id,
                               eos_token_id=tok.eos_token_id)
            for r, s in zip(b, tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                                skip_special_tokens=True)):
                out[split].append(dict(instruction=r["instruction"], teacher=s.strip(),
                                       alpaca=r["alpaca"]))
            print(f"{split} {len(out[split])}/{len(rows)} {time.time() - t0:.0f}s",
                  flush=True)
            tmp = PARTIAL.with_suffix(".json.tmp")
            json.dump(out, open(tmp, "w"), ensure_ascii=False)
            os.replace(tmp, PARTIAL)
    os.replace(PARTIAL, OUT)
    print("teacher data done", OUT, flush=True)


if __name__ == "__main__":
    main()
