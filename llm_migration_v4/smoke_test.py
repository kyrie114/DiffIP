"""P0 smoke test: three quick checks before launching the long jobs (PLAN.md §3).

Usage: python smoke_test.py [s1|s2|s3|all]
  S1 collect : 1B model, 1 prompt, 1 seed set, N=128, 4 layers
               -> shape [4, 128, 2048], no NaN, deterministic (bit-identical rerun)
  S2 teacher : 3B model generates 1 answer with the built-in chat template -> non-empty
  S3 student : 1B model (fp32), chat-template format, 2 training steps on 1 example
               -> finite loss
"""
import sys
import time

import numpy as np
import torch

from common import (H, MODELS, PROMPTS, N_LAYERS, N_MAX, SEED_SETS,
                    even_layers, load, load_tok, perturb_batch, pooled_layers)


def s1():
    torch.set_num_threads(16)
    path = str(MODELS / "Llama-3.2-1B")
    m, tok = load(path)
    total = len(m.model.layers)
    L4 = even_layers(total, N_LAYERS)
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids
    base = m.get_input_embeddings()(ids).detach()  # [1, T, D]
    x = perturb_batch(base, N_MAX, SEED_SETS[0])
    a = pooled_layers(m, x, L4)
    b = pooled_layers(m, x, L4)
    assert a.shape == (N_LAYERS, N_MAX, m.config.hidden_size), a.shape
    assert not np.isnan(a).any(), "NaN in pooled features"
    assert np.array_equal(a, b), "pooled_layers not deterministic"
    print(f"S1 PASS  shape={a.shape} layers={L4} mean_abs={np.abs(a).mean():.4f}")
    del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def s2():
    torch.set_num_threads(16)
    path = str(MODELS / "Llama-3.2-3B")
    m, tok = load(path)
    p = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
    enc = tok(p, return_tensors="pt")
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=48, do_sample=False,
                       pad_token_id=tok.pad_token_id,
                       eos_token_id=tok.eos_token_id)
    ans = tok.decode(g[0, enc["input_ids"].shape[1]:],
                     skip_special_tokens=True).strip()
    assert len(ans) > 0, "empty teacher answer"
    print(f"S2 PASS  answer={ans[:60]!r}")


def s3():
    torch.set_num_threads(12)
    import json
    from transformers import AutoModelForCausalLM
    path = str(MODELS / "Llama-3.2-1B")
    tok = load_tok(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    D = json.load(open(H / "data" / "teacher_data.json"))["train"]
    p = tok(f"### Instruction:\n{D[0]['instruction']}\n\n### Response:\n")["input_ids"]
    a = tok(D[0]["teacher"] + tok.eos_token)["input_ids"]
    s_ = (p + a)[:320]
    ii = torch.tensor([s_])
    am = torch.ones_like(ii)
    lab = torch.full_like(ii, -100)
    lab[0, len(p):] = torch.tensor(s_[len(p):])
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    losses = []
    for step in range(2):
        loss = m(input_ids=ii, attention_mask=am, labels=lab).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss))
        print(f"  step{step} loss={losses[-1]:.4f}", flush=True)
    assert all(np.isfinite(losses)), "non-finite loss"
    print(f"S3 PASS  losses={losses}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("s1", "all"):
        s1()
    if which in ("s2", "all"):
        s2()
    if which in ("s3", "all"):
        s3()
    print(f"smoke done in {time.time() - t0:.0f}s")
