"""Diagnostic: isolate the transformers-version RoPE bug on local models.

Stage 1 (diffip-llm, transformers 5.20.2):
  - Llama-3.2-1B batch=1 inputs_embeds forward, dump q/k/cos/sin shapes at first attn layer
  - Llama-3.2-1B batch=1 with explicit position_ids
  - Llama-2-7b-hf batch=64 inputs_embeds forward (main-line viability)
Stage 2 (robust-fp, transformers 4.44.2):
  - architecture registration check for all v4 models (config only, no weights)
  - Llama-3.2-1B batch=64 inputs_embeds forward (does 4.44 work?)

Run: python diag_rope.py <env-python-path>   (run twice, once per env)
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODELS = "/data/luo_lixuan/llms"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "5.20"
sys.argv = [sys.argv[0]]  # keep common import clean
sys.path.insert(0, "/home/lushunxin/work/DiffIP/llm_migration_v4")
from common import N_LAYERS, SCALE, SEED_SETS, even_layers, load, perturb_batch, pooled_layers  # noqa: E402

torch.set_num_threads(16)


def stage1():
    print("transformers", __import__("transformers").__version__, flush=True)
    # 1a: Llama-3.2-1B batch=1 inputs_embeds
    try:
        m, tok = load(f"{MODELS}/Llama-3.2-1B")
        ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids
        T = ids.shape[1]
        base = m.get_input_embeddings()(ids).detach()
        shapes = {}

        def hook(mod, inp, out):
            if "q" not in shapes:
                # recompute q shape from weights
                h = inp[0]
                q = mod.q_proj(h)
                B, Tp, _ = q.shape
                H = m.config.num_attention_heads
                shapes["q"] = (B, H, Tp, m.config.head_dim)
                shapes["k"] = (B, m.config.num_key_value_heads, Tp, m.config.head_dim)
                shapes["config_heads"] = (H, m.config.num_key_value_heads, m.config.head_dim)

        h1 = m.model.layers[0].self_attn.register_forward_hook(hook)
        with torch.no_grad():
            m(inputs_embeds=base.unsqueeze(0),
              attention_mask=torch.ones(1, T, dtype=torch.long),
              use_cache=False, output_hidden_states=True)
        h1.remove()
        print("3.2-1B b1 inputs_embeds: OK", shapes, flush=True)
    except Exception as e:
        print("3.2-1B b1 inputs_embeds: FAIL", type(e).__name__, str(e)[:120], flush=True)
    # 1b: explicit position_ids
    try:
        pos = torch.arange(T).unsqueeze(0)
        with torch.no_grad():
            m(inputs_embeds=base.unsqueeze(0),
              attention_mask=torch.ones(1, T, dtype=torch.long),
              position_ids=pos, use_cache=False)
        print("3.2-1B b1 explicit-pos: OK", flush=True)
    except Exception as e:
        print("3.2-1B b1 explicit-pos: FAIL", type(e).__name__, str(e)[:120], flush=True)
    del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # 1c: Llama-2-7b-hf batch=64 inputs_embeds (main line)
    try:
        m2, tok2 = load(f"{MODELS}/Llama-2-7b-hf")
        ids2 = tok2("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids
        T2 = ids2.shape[1]
        base2 = m2.get_input_embeddings()(ids2).detach()
        x = perturb_batch(base2, 64, SEED_SETS[0])
        r = pooled_layers(m2, x, even_layers(m2.config.num_hidden_layers, N_LAYERS))
        import numpy as np
        print("llama2-7b b64 inputs_embeds: OK", r.shape,
              "nan" if np.isnan(r).any() else "no-nan", flush=True)
    except Exception as e:
        print("llama2-7b b64 inputs_embeds: FAIL", type(e).__name__, str(e)[:120], flush=True)


def stage2():
    print("transformers", __import__("transformers").__version__, flush=True)
    ok = True
    for name in ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "Chinese-llama-2-7b",
                 "llama2-7B-pruned-sparsegpt", "wanda_pruned_llama_2_7B",
                 "GBLM-Pruner-LLaMA-2-7B", "Qwen2.5-7B", "Meta-Llama-3-8B",
                 "internlm2-7b", "Llama-3.2-1B", "Llama-3.2-3B", "Sheared-LLaMA-1.3B"]:
        try:
            c = AutoConfig.from_pretrained(f"{MODELS}/{name}")
            AutoModelForCausalLM.from_config(c)
            print(f"  arch OK   {name} ({c.model_type})", flush=True)
        except Exception as e:
            ok = False
            print(f"  arch FAIL {name}: {type(e).__name__} {str(e)[:80]}", flush=True)
    try:
        m, tok = load(f"{MODELS}/Llama-3.2-1B")
        ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids
        T = ids.shape[1]
        base = m.get_input_embeddings()(ids).detach()
        x = perturb_batch(base, 64, SEED_SETS[0])
        r = pooled_layers(m, x, even_layers(m.config.num_hidden_layers, N_LAYERS))
        import numpy as np
        print("3.2-1B b64 inputs_embeds:", "OK" if not np.isnan(r).any() else "NAN",
              r.shape, flush=True)
    except Exception as e:
        print("3.2-1B b64 inputs_embeds: FAIL", type(e).__name__, str(e)[:120], flush=True)
    return ok


if __name__ == "__main__":
    which = WHICH
    if which == "5.20":
        stage1()
    elif which == "4.44":
        ok = stage2()
        sys.exit(0 if ok else 1)
