"""Unrelated-model negative panel vs victim Qwen/Qwen2.5-0.5B.
Same protocol as run.py: LLMRepresentationCollector, perturbation, seed 123, N=8,
scale 2e-3, L=4 evenly spaced, DP off, ALS mean L2, sim=exp(-sum d/L); dims 8 and 20.
Non-Llama-style archs (GPT2: transformer.h, Pythia: gpt_neox.layers) are exposed to the
collector via a model.model.layers alias; pooling/noise logic is unchanged."""
import gc, json, sys, types
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent)); sys.path.insert(0, str(H))
from diffip.llm import LLMRepresentationCollector
from score_als import score_pair
PROMPT="The quick brown fox jumps over the lazy dog."
NEGS=[("smollm2_135m","HuggingFaceTB/SmolLM2-135M","unrelated"),
      ("smollm2_360m","HuggingFaceTB/SmolLM2-360M","unrelated"),
      ("gpt2","openai-community/gpt2","unrelated"),
      ("pythia_160m","EleutherAI/pythia-160m","unrelated"),
      ("qwen2_0.5b","Qwen/Qwen2-0.5B","same_family_different_model")]
FN=H/"feats_neg"
def even(t,c): return [round(i*(t-1)/(c-1)) for i in range(c)]
def collect(mid):
    tok=AutoTokenizer.from_pretrained(mid); m=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.float32).eval()
    if not hasattr(m,"model"):
        layers = m.transformer.h if hasattr(m,"transformer") else m.gpt_neox.layers
        m.model=types.SimpleNamespace(layers=layers)
    L=even(m.config.num_hidden_layers,4)
    r=LLMRepresentationCollector(m,tok,n_samples=8,perturbation_scale=2e-3,layers=L,seed=123,sample_mode="perturbation").collect(PROMPT).representations
    del m; gc.collect(); return r,L
victim=np.load(H/"feats/victim.npy")
cat,rows={},[]
for name,mid,kind in NEGS:
    p=FN/f"{name}.npy"
    if p.exists(): r=np.load(p); L="cached"
    else: r,L=collect(mid); np.save(p,r.astype(np.float64))
    cat[name]={"model_id":mid,"kind":kind,"layers":L,"shape":list(r.shape)}
    row={"neg":name,"model_id":mid,"kind":kind}
    for d in (8,20):
        a=score_pair(r,victim,n_components=d,also_full_metric=False)["als"]
        row[f"d{d}"]={"similarity":a["similarity"],"distance_mean":a["distance_mean"],"n_components_used":a["n_components"]}
    rows.append(row)
    print(f"{name:<14}{kind:<30} 8: {row['d8']['similarity']:.4f}/{row['d8']['distance_mean']:.4f}  20: {row['d20']['similarity']:.4f}/{row['d20']['distance_mean']:.4f}",flush=True)
der=json.loads((H/"results_dim8.json").read_text())["results"]
def stats(kind_filter,d,key):
    v=[r[f"d{d}"][key] for r in rows if r["kind"] in kind_filter]
    return {"min":min(v),"mean":float(np.mean(v)),"max":max(v),"n":len(v)}
summary={}
for d in (8,20):
    summary[d]={"unrelated":{k:stats({"unrelated"},d,k) for k in ("similarity","distance_mean")},
                "all_incl_same_family":{k:stats({"unrelated","same_family_different_model"},d,k) for k in ("similarity","distance_mean")}}
side=[]
for r in der:
    if r["victim"]!="victim" or r["pair"] in ("self","neg_cross_family"): continue
    e={"pair":r["pair"]}
    for d in (8,20):
        s=r[f"d{d}"]["similarity"]; u=summary[d]["unrelated"]["similarity"]; a=summary[d]["all_incl_same_family"]["similarity"]
        e[f"d{d}"]={"similarity":s,"distance_mean":r[f"d{d}"]["distance_mean"],
                    "above_max_unrelated":s>u["max"],"above_max_incl_same_family":s>a["max"]}
    side.append(e)
out={"victim":"Qwen/Qwen2.5-0.5B (feats/victim.npy, same as v2)","protocol":"see run.py; dims 8 and 20; DP off",
     "held_out_note":"dim 8 chosen in pca_sweep using only SmolLM2-135M negative; other negatives are held-out",
     "catalog":cat,"negatives":rows,"negative_summary":summary,"derived_vs_victim":side}
(H/"results_neg_panel.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
for d in (8,20): print(d,json.dumps(summary[d]))
for e in side: print(e)
