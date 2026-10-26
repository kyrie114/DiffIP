"""Shared pieces for llm_migration_v3 (stricter LLN rerun).
Perturbation collector here is a *batched* re-implementation of
diffip.llm.LLMRepresentationCollector(sample_mode='perturbation'): sample 0 is the
clean embedding, samples 1..N-1 add scale*randn drawn sequentially from
torch.Generator().manual_seed(seed) with the embedding's shape; attention-mask mean
pooling of decoder-layer outputs. Batching does not change the math (equal-length
copies of one prompt, no padding); test_equivalence.py checks this numerically."""
from __future__ import annotations
import gc, sys, types
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
H = Path(__file__).resolve().parent
ROOT = H.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "llm_migration_v2"))
from transforms import apply_linear_residual_transform, make_permutation, make_anisotropic_scale, prune_layers  # noqa
torch.set_num_threads(8)
SCALE = 2e-3
SEED_SETS = [123, 1123, 2123]
N_MAX = 128
N_GRID = [8, 16, 32, 64, 128]
VICTIM = "Qwen/Qwen2.5-0.5B"
INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"
NEGS = {"neg_smollm2_135m": "HuggingFaceTB/SmolLM2-135M",
        "neg_smollm2_360m": "HuggingFaceTB/SmolLM2-360M",
        "neg_gpt2": "openai-community/gpt2",
        "neg_pythia_160m": "EleutherAI/pythia-160m",
        "neg_qwen2_0.5b": "Qwen/Qwen2-0.5B"}  # last one = same family, different model
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
 "git commit -m \"fix: handle empty input\"",
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
CATEGORY = ["en"] * 7 + ["zh"] * 6 + ["code"] * 6 + ["math"] * 6 + ["qa"] * 7

def even_layers(total, count):
    return [0] if count == 1 else [round(i * (total - 1) / (count - 1)) for i in range(count)]

def decoder_layers(m):
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer"):
        return m.transformer.h
    return m.gpt_neox.layers

def load(mid):
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    return m, tok

def prune_spec(total):
    n_drop = 4 if total > 8 else max(1, total // 6)
    return [round(i * (total - 1) / (n_drop + 1)) for i in range(1, n_drop + 1)]

def build_model(name):
    """Returns (model, tok, embed_fn) where embed_fn(input_ids)->inputs_embeds fed to the model."""
    if name == "victim": m, t = load(VICTIM)
    elif name in ("instruct", "instruct_wperm", "instruct_prune"): m, t = load(INSTRUCT)
    else: m, t = load(NEGS[name])
    E = m.get_input_embeddings()
    if name == "instruct_wperm":
        with torch.no_grad():
            clean = E.weight.detach().clone()
        P = make_permutation(m.config.hidden_size, seed=0)
        apply_linear_residual_transform(m, P, fold=False)
        # paired inputs: noise is added in the *original* basis then mapped by P,
        # i.e. inputs = (E_clean[x] + eps) @ P  (same as v2 collect_shared)
        return m, t, (lambda ids: clean[ids]), P
    if name == "instruct_prune":
        prune_layers(m, prune_spec(m.config.num_hidden_layers))
    return m, t, (lambda ids: E(ids)), None

@torch.no_grad()
def perturb_batch(base, n, seed):
    """[n, T, D] = clean + (n-1) perturbed, noise drawn sequentially exactly like the collector."""
    g = torch.Generator().manual_seed(seed)
    out = [base]
    for _ in range(n - 1):
        out.append(base + SCALE * torch.randn(base.shape, generator=g, dtype=base.dtype))
    return torch.cat(out, 0)

@torch.no_grad()
def forward_pool(m, embeds, layers, chunk=64):
    """embeds [B,T,D] -> pooled [len(layers), B, D] (all-ones mask: single prompt, no pad)."""
    dec = decoder_layers(m); hid = {}
    hooks = [dec[i].register_forward_hook(
        (lambda i: lambda _m, _in, o: hid.__setitem__(i, (o[0] if isinstance(o, tuple) else o)))(i))
        for i in layers]
    res = [[] for _ in layers]
    try:
        for s in range(0, embeds.shape[0], chunk):
            e = embeds[s:s + chunk]
            m(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], dtype=torch.long), use_cache=False)
            for k, i in enumerate(layers):
                res[k].append(hid[i].mean(dim=1).to(torch.float32))
    finally:
        for h in hooks: h.remove()
    return torch.stack([torch.cat(r, 0) for r in res], 0).numpy()
