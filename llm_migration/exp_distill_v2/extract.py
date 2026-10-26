"""步骤4：72 条指纹提示（与方法A/旋转攻击相同）逐层 last-token hidden，CPU fp32，与 exp_rotation_attack/extract.py 相同抽取方式。"""
import json, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(8); torch.set_grad_enabled(False)
_P = json.load(open("/workspace/20261019_DiffIP迁移LLM_线性回拉/prompts.json")); PROMPTS = _P["calibration"] + _P["test"]
M = {"teacher": "Qwen/Qwen2.5-0.5B-Instruct", "student_distill": "ckpt/student_distill", "student_control": "ckpt/student_control",
     "smollm135_base": "HuggingFaceTB/SmolLM2-135M", "smollm360": "HuggingFaceTB/SmolLM2-360M", "gpt2": "openai-community/gpt2",
     "pythia160": "EleutherAI/pythia-160m", "qwen2_05b": "Qwen/Qwen2-0.5B"}
for name in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(M[name]); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    m = AutoModelForCausalLM.from_pretrained(M[name], dtype=torch.float32).eval(); rows = []
    for i in range(0, 72, 8):
        enc = tok(PROMPTS[i:i+8], return_tensors="pt", padding=True)
        Hs = torch.stack(m(**enc, output_hidden_states=True).hidden_states, 1); lens = enc["attention_mask"].sum(1) - 1
        rows += [Hs[b, :, lens[b], :].numpy() for b in range(Hs.shape[0])]
    X = np.stack(rows).astype(np.float32); np.save(f"feats/{name}.npy", X); print(name, X.shape, flush=True); del m
