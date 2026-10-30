"""Shared pieces for llm_migration_v4 (7B-scale rerun of the v3 protocol on local models).

Protocol identical to llm_migration_v3 scheme (a): paired-seed input-embedding
perturbation (SCALE 2e-3), 32 prompts, 3 seed sets, N_MAX=128, 4 even layers,
attention-mask mean pooling, PCA + per-layer ALS (no DP).

Differences from v3:
- All models from the local library /data/luo_lixuan/llms (no downloads).
- Main line = Llama-2-7b family: victim Llama-2-7b-hf, official chat SFT as the
  fine_tune positive, Chinese-llama-2-7b as an independent real derivative,
  same-base pruned variants as hard negatives, cross-family 7B/8B as easy ones.
- Distill line: Llama-3.2-1B victim, students distilled from Llama-3.2-3B.
- Collection via output_hidden_states (no hooks) with a batch=1 fallback
  (transformers-version dependent batched-forward issue seen once in 5.20).
"""
from __future__ import annotations
import gc
import sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

H = Path(__file__).resolve().parent
ROOT = H.parent
MODELS = Path("/data/luo_lixuan/llms")
sys.path.insert(0, str(ROOT / "llm_migration_v2"))
from transforms import apply_linear_residual_transform, make_permutation, make_anisotropic_scale, prune_layers  # noqa: E402

# Local model copies ship without chat_template in tokenizer_config.json;
# use the official Llama-3.2 template string so generation/training work.
LLAMA32_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\n\n' }}{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


def load_tok(mid):
    tok = AutoTokenizer.from_pretrained(mid)
    if not getattr(tok, "chat_template", None):
        tok.chat_template = LLAMA32_CHAT_TEMPLATE
    return tok

SCALE = 2e-3
SEED_SETS = [123, 1123, 2123]
N_MAX = 128
N_GRID = [8, 16, 20, 32, 64, 128]
DIMS = [8, 20]
N_LAYERS = 4

PROMPTS = [
 # English
 "The quick brown fox jumps over the lazy dog.",
 "Once upon a time, in a small village by the sea, there lived an old fisherman.",
 "Climate change is one of the most pressing challenges facing humanity today.",
 "Please write a short email to my manager asking for a day off next Friday.",
 "The Industrial Revolution began in Britain in the late eighteenth century.",
 "My favorite hobby is hiking in the mountains during autumn.",
 "Explain why the sky appears blue during the day.",
 # Chinese
 "床前明月光，疑是地上霜。",
 "今天天气很好，我们一起去公园散步吧。",
 "人工智能正在深刻地改变我们的生活方式。",
 "请用三句话介绍一下长城的历史。",
 "北京是中华人民共和国的首都。",
 "学而时习之，不亦说乎？",
 # code
 "def fibonacci(n):\n    if n < 2:\n        return n\n",
 "import numpy as np\nx = np.linspace(0, 1, 100)\n",
 "SELECT name, age FROM users WHERE age > 30 ORDER BY age;",
 "for (int i = 0; i < n; i++) { sum += a[i]; }",
 "class Stack:\n    def __init__(self):\n        self.items = []\n",
 'git commit -m "fix: handle empty input"',
 # math
 "What is 17 multiplied by 23?",
 "Solve for x: 2x + 5 = 13.",
 "The derivative of sin(x) is cos(x).",
 "Prove that the square root of 2 is irrational.",
 "If a triangle has sides 3, 4 and 5, what is its area?",
 "The sum of the first n positive integers is n(n+1)/2.",
 # QA
 "Q: What is the capital of France?\nA:",
 "Q: Who wrote Romeo and Juliet?\nA:",
 "Q: How many legs does a spider have?\nA:",
 "Q: What is the boiling point of water at sea level?\nA:",
 "Q: Which planet is known as the Red Planet?\nA:",
 "问：水的化学式是什么？\n答：",
 "Q: What year did World War II end?\nA:",
]
assert len(PROMPTS) == 32

# ---- main line (Llama-2-7b family) ----
MAIN = {
    "victim": f"{MODELS}/Llama-2-7b-hf",
    "fine_tune": f"{MODELS}/Llama-2-7b-chat-hf",
    "derivative_cn": f"{MODELS}/Chinese-llama-2-7b",
    "weight_perm": f"{MODELS}/Llama-2-7b-chat-hf",   # + permutation disguise
    "prune_dpoff": f"{MODELS}/Llama-2-7b-chat-hf",   # + 4-layer prune
    "neg_pruned_sparsegpt": f"{MODELS}/llama2-7B-pruned-sparsegpt",
    "neg_pruned_wanda": f"{MODELS}/wanda_pruned_llama_2_7B",
    "neg_pruned_gblm": f"{MODELS}/GBLM-Pruner-LLaMA-2-7B",
    "neg_qwen2_5_7b": f"{MODELS}/Qwen2.5-7B",
    "neg_llama3_8b": f"{MODELS}/Meta-Llama-3-8B",
    "neg_internlm2_7b": f"{MODELS}/internlm2-7b",
}
# rep_perm / scale are derived from fine_tune features in score.py (no collection)

# ---- distill line (Llama-3.2) ----
DISTILL = {
    "victim": f"{MODELS}/Llama-3.2-1B",
    "student_distill": str(H / "ckpt" / "student_distill"),
    "student_control": str(H / "ckpt" / "student_control"),
    "neg_llama3_2_3b": f"{MODELS}/Llama-3.2-3B",
    "neg_sheared_1_3b": f"{MODELS}/Sheared-LLaMA-1.3B",
    "neg_qwen2_5_7b": f"{MODELS}/Qwen2.5-7B",
    "neg_llama2_7b": f"{MODELS}/Llama-2-7b-hf",   # cross-D (4096 vs 2048)
}


def even_layers(total, count):
    return [0] if count == 1 else [round(i * (total - 1) / (count - 1)) for i in range(count)]


def prune_spec(total):
    n_drop = 4 if total > 8 else max(1, total // 6)
    return [round(i * (total - 1) / (n_drop + 1)) for i in range(1, n_drop + 1)]


def load(mid):
    tok = load_tok(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True).eval()
    return m, tok


@torch.no_grad()
def perturb_batch(base, n, seed):
    """[n, T, D] = clean + (n-1) perturbed, noise drawn sequentially exactly like the collector."""
    g = torch.Generator().manual_seed(seed)
    out = [base]
    for _ in range(n - 1):
        out.append(base + SCALE * torch.randn(base.shape, generator=g, dtype=base.dtype))
    return torch.cat(out, 0)


@torch.no_grad()
def pooled_layers(model, embeds, layers, chunk=64):
    """[n, T, D] -> [len(layers), n, D] via output_hidden_states (no hooks).

    Falls back to batch=1 if the batched forward fails."""
    am = torch.ones(embeds.shape[:2], dtype=torch.long)
    try:
        with torch.no_grad():
            out = model(inputs_embeds=embeds, attention_mask=am,
                        use_cache=False, output_hidden_states=True)
        hs = out.hidden_states  # [embed_out, layer0, ..., layer_{L-1}]
        picked = torch.stack([hs[i + 1] for i in layers], 0)  # [L, n, T, D]
        return picked.mean(dim=2).to(torch.float32).numpy()
    except RuntimeError as e:
        print(f"  batched forward failed ({str(e)[:60]}); falling back to batch=1",
              flush=True)
        res = [[] for _ in layers]
        for s in range(embeds.shape[0]):
            with torch.no_grad():
                out = model(inputs_embeds=embeds[s:s + 1], attention_mask=am[s:s + 1],
                            use_cache=False, output_hidden_states=True)
            hs = out.hidden_states
            for k, i in enumerate(layers):
                res[k].append(hs[i + 1][0].mean(dim=0))
        return torch.stack([torch.stack(r, 0) for r in res], 0).to(torch.float32).numpy()


def build_main(name):
    """Returns (model, tok, embed_fn, perm_idx) for a main-line model.

    embed_fn(input_ids) -> inputs_embeds; perm_idx (or None) is the index
    permutation to apply to perturbed embeddings (weight_perm disguise only)."""
    if name == "weight_perm":
        m, t = load(MAIN[name])
        E = m.get_input_embeddings()
        clean = E.weight.detach().clone()
        P = make_permutation(m.config.hidden_size, seed=0)
        apply_linear_residual_transform(m, P, fold=False)
        # paired inputs: noise added in the *original* basis then mapped,
        # x @ P == x[..., P.argmax(0)]  (same as v3 collect_perturb)
        return m, t, (lambda ids: clean[ids]), P.argmax(0).numpy()
    if name == "prune_dpoff":
        m, t = load(MAIN[name])
        prune_layers(m, prune_spec(m.config.num_hidden_layers))
        E = m.get_input_embeddings()
        return m, t, (lambda ids: E(ids)), None
    m, t = load(MAIN[name])
    E = m.get_input_embeddings()
    return m, t, (lambda ids: E(ids)), None
