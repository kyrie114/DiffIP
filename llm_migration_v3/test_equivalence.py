"""Batched collector == official LLMRepresentationCollector (victim, 1 prompt, N=8)."""
from common import *
from diffip.llm import LLMRepresentationCollector
m, t, emb, _ = build_model("victim")
L = even_layers(m.config.num_hidden_layers, 4); p = PROMPTS[0]
off = LLMRepresentationCollector(m, t, n_samples=8, perturbation_scale=SCALE, layers=L, seed=123,
                                 sample_mode="perturbation").collect(p).representations
ids = t(p, return_tensors="pt")["input_ids"]
mine = forward_pool(m, perturb_batch(emb(ids), 8, 123), L)
v2 = np.load(ROOT / "llm_migration_v2/feats/victim.npy")
print("max|official-batched|", np.abs(off - mine).max(), " max|v2 cached - batched|", np.abs(v2 - mine).max(), " scale", np.abs(off).mean())
