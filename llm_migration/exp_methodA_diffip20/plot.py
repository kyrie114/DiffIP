import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "NotoSansCJK" in f: fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
T = json.load(open("results.json"))["table"]; P = json.load(open("results.json"))["per_subset"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
order = ["cos", "pca5", "pca10", "pca20", "pca35", "full"]
for k, (st, vs) in enumerate([("S1", ["all"]), ("S2", ["no_student", "with_student"])]):
    for j, v in enumerate(vs):
        ms = [m for m in order if any(t["setting"] == st and t["variant"] == v and t["method"] == m for t in T)]
        for i, m in enumerate(ms):
            ys = [r["margin"] for r in P if r["setting"] == st and r["variant"] == v and r["method"] == m]
            ax[k].scatter([i + 0.15 * (j - 0.5 * (len(vs) - 1))] * len(ys), ys, s=18, c=["tab:blue", "tab:red"][j],
                          label=v if i == 0 else None)
        ax[k].set_xticks(range(len(ms))); ax[k].set_xticklabels(ms)
    ax[k].axhline(0, c="k", lw=0.8); ax[k].set_ylabel("margin = min正 − max负（每个点=一个提示子集）")
    ax[k].legend(fontsize=8)
ax[0].set_title("S1 复现 10/19：Qwen2.5-0.5B；正=Instruct,LoRA；负=Qwen2-0.5B"); ax[0].set_yscale("symlog", linthresh=0.01)
ax[1].set_title("S2 复用 10/16：正=Instruct,int8,prune30(+GPT2学生)；负=4个异源")
fig.tight_layout(); fig.savefig("/workspace/papers/figs/methodA_diffip20_margin.png", dpi=150)
