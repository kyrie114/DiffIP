"""Rescore all v2 pairs at n_components=8 (chosen via pca_sweep) and 20 (official default).
Uses the cached v2 feats produced by run.py (same protocol: paired seed 123, N=8,
scale 2e-3, L=4). DP off, ALS mean L2, similarity=exp(-sum d/L)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from score_als import score_pair
F = HERE / "feats"
pairs = [
 ("self","victim","victim","baseline"),
 ("fine_tune_instruct","instruct","victim","A_finetune"),
 ("neg_cross_family","neg_smol","victim","negative"),
 ("perm_vs_instruct_shared","instruct_perm","instruct_shared","B_weight_perm"),
 ("perm_vs_victim","instruct_perm","victim","B_weight_perm"),
 ("feat_perm_vs_instruct","instruct_feat_perm","instruct_shared","B_repr_perm"),
 ("scale_vs_instruct","instruct_scale","instruct_shared","C_scaling"),
 ("scale_vs_victim","instruct_scale","victim","C_scaling"),
 ("prune_vs_instruct","instruct_prune","instruct","D_pruning"),
 ("prune_vs_victim","instruct_prune","victim","D_pruning"),
]
rows=[]
for name,s,v,g in pairs:
    S,V=np.load(F/f"{s}.npy"),np.load(F/f"{v}.npy")
    r={"pair":name,"group":g,"suspect":s,"victim":v}
    for d in (8,20):
        a=score_pair(S,V,n_components=d,also_full_metric=False)["als"]
        r[f"d{d}"]={"similarity":a["similarity"],"distance_mean":a["distance_mean"],
                    "per_layer_distance":a["per_layer_distance"],"n_components_used":a["n_components"],
                    "converged_all":a["converged_all"]}
    rows.append(r)
    print(f"{name:<26} 8: sim={r['d8']['similarity']:.4f} d={r['d8']['distance_mean']:.4f} | 20: sim={r['d20']['similarity']:.4f} d={r['d20']['distance_mean']:.4f}")
out={"protocol":{"n_components_primary":8,"n_components_reference":20,"chosen_from":"pca_sweep/维数扫描.md",
     "seed":123,"n_samples":8,"perturbation_scale":2e-3,"n_layers":4,"dp":"off","metric":"ALS mean L2; sim=exp(-sum d/L)",
     "lora":"no LoRA checkpoint in v2; not run"},"results":rows}
(HERE/"results_dim8.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
