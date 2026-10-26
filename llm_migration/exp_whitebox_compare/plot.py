import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
T = json.load(open("results.json"))["table"]
groups = [("S1", "all", "S1 微调/LoRA"), ("S2", "no_student", "S2 无学生"), ("S2", "with_student", "S2 含蒸馏学生")]
meths = ["cos", "REEF", "REEF_avg4", "DiffIP20"]; fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for k, key in enumerate(["margin_mean", "rel_margin_mean"]):
    for i, (s, v, name) in enumerate(groups):
        for j, m in enumerate(meths):
            r = next(t for t in T if t["setting"] == s and t["variant"] == v and t["method"] == m)
            if key not in r: ax[k].text(i + (j - 1.5) * .2, 0, "n/a", ha="center", fontsize=7); continue
            ax[k].bar(i + (j - 1.5) * .2, r[key], .2, color=f"C{j}", label=m if i == 0 else None)
            if key == "margin_mean": ax[k].errorbar(i + (j - 1.5) * .2, r[key], [[r[key] - r["margin_min"]], [r["margin_max"] - r[key]]], color="k", capsize=2)
    ax[k].axhline(0, color="k", lw=.8); ax[k].set_xticks(range(3)); ax[k].set_xticklabels([g[2] for g in groups])
    ax[k].set_title("margin = min正 − max负（误差棒=子集 min/max）" if k == 0 else "相对 margin = margin / 子集内分数极差"); ax[k].legend(fontsize=8)
ax[0].set_yscale("symlog", linthresh=0.05)
plt.tight_layout(); plt.savefig("/workspace/papers/figs/whitebox_compare_margin.png", dpi=150)
