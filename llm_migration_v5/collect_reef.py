"""干净前向，存每一层最后一个 token。规则见 试验方案.md 第二节。

不写 feats/。输出本目录 acts_reef/main_{name}.npz，last [32, layers, D]。
已有文件跳过，便于从 victim 之后接着采。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
V4 = H.parent / "llm_migration_v4"
sys.path.insert(0, str(V4))
from collect import n_decoder_layers  # noqa: E402
from common import PROMPTS, QWEN3_ROUND, UNRELATED_ROUND, build_main  # noqa: E402

OUT = H / "acts_reef"
OUT.mkdir(exist_ok=True)
NAMES = [
    "victim", "fine_tune", "weight_perm", "prune_dpoff", "derivative_cn",
    "neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm",
    "neg_qwen2_5_7b", "neg_llama3_8b", "neg_llama1_7b", "neg_baichuan_7b", "neg_mpt_7b",
]


def collect_one(name):
    import gc
    import torch
    torch.set_num_threads(int(os.environ.get("V4_THREADS", "8")))
    out = OUT / f"main_{name}.npz"
    if out.exists():
        print("skip", name, flush=True)
        return
    t0 = time.time()
    model, tok, emb, perm_idx = build_main(name)
    total = n_decoder_layers(model)
    D = model.get_input_embeddings().weight.shape[1]
    last = np.zeros((len(PROMPTS), total, D), dtype=np.float32)
    for p, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids
        x = emb(ids).detach()
        if perm_idx is not None:
            x = x[..., perm_idx]
        am = torch.ones(x.shape[:2], dtype=torch.long)
        with torch.no_grad():
            hs = model(inputs_embeds=x, attention_mask=am, use_cache=False,
                       output_hidden_states=True, return_dict=True).hidden_states
        for i in range(total):
            last[p, i] = hs[i + 1][0, -1].float().cpu().numpy()
        if not np.isfinite(last[p]).all():
            bad = last[p]
            raise SystemExit(
                f"{name} prompt {p + 1}: non-finite last token "
                f"nan={int(np.isnan(bad).sum())} inf={int(np.isinf(bad).sum())}")
        print(f"  {name} prompt {p + 1}/{len(PROMPTS)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(out, last=last, total=np.int64(total))
    print(f"saved {out.name} layers={total} D={D} {time.time() - t0:.0f}s", flush=True)
    del model
    gc.collect()


def main():
    names = sys.argv[1:] or NAMES
    known = set(NAMES) | set(UNRELATED_ROUND) | set(QWEN3_ROUND)
    unknown = [n for n in names if n not in known and n not in NAMES]
    if unknown:
        raise SystemExit(f"unknown: {unknown}")
    # One process per model. A long-lived process that had already loaded the
    # Llama and Qwen checkpoints produced an all-NaN Baichuan forward on a
    # prompt that is finite in a fresh process.
    if os.environ.get("V4_REEF_WORKER") == "1":
        for name in names:
            collect_one(name)
        return
    import subprocess
    env = os.environ.copy()
    env["V4_REEF_WORKER"] = "1"
    for name in names:
        if (OUT / f"main_{name}.npz").exists():
            print("skip", name, flush=True)
            continue
        proc = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), name], env=env)
        if proc.returncode != 0:
            raise SystemExit(f"{name} exited {proc.returncode}")


if __name__ == "__main__":
    main()
