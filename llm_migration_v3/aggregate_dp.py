import json, numpy as np
from common import H, NEGS
names = ["prune", "fine_tune", *NEGS]; L = ["\n| suspect (vs victim) | dims | Ls×Lv | DP-on sim mean±std (min) | DP-on dist/√(LsLv) | DP-off sim mean±std (min) | DP-off dist mean |", "|---|---|---|---|---|---|---|"]
out = []
for n in names:
    rows = json.loads((H / "scores" / f"dp_{n}.json").read_text())
    for d in [8, 20]:
        r = [x for x in rows if x["dims"] == d]; a = lambda k: np.array([x[k] for x in r])
        o = dict(pair=n, dims=d, Ls=r[0]["Ls"], Lv=r[0]["Lv"], n_prompts=len(r),
                 on_sim=a("dp_on_similarity").mean(), on_sim_std=a("dp_on_similarity").std(ddof=1), on_sim_min=a("dp_on_similarity").min(),
                 on_dist=a("dp_on_distance_per_step").mean(), off_sim=a("dp_off_similarity").mean(),
                 off_sim_std=a("dp_off_similarity").std(ddof=1), off_sim_min=a("dp_off_similarity").min(), off_dist=a("dp_off_distance_mean").mean())
        out.append(o)
        L.append(f"| {n} | {d} | {o['Ls']}×{o['Lv']} | {o['on_sim']:.3f}±{o['on_sim_std']:.3f} ({o['on_sim_min']:.3f}) | {o['on_dist']:.3f} | {o['off_sim']:.3f}±{o['off_sim_std']:.3f} ({o['off_sim_min']:.3f}) | {o['off_dist']:.3f} |")
(H / "summary_dp.json").write_text(json.dumps(out, indent=1, default=float)); (H / "tables_dp.md").write_text("\n".join(L)); print("\n".join(L))
