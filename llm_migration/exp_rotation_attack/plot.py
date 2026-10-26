import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
R = [r for r in json.load(open("results.json"))["rows"] if r["subset"] == "all72"]
ms = ["cos", "REEF_cka_std", "REEF_cka_plain", "DiffIP20"]; mods = [r["model"] for r in R]
fig, ax = plt.subplots(figsize=(10, 4))
for j, m in enumerate(ms): ax.bar(np.arange(len(R)) + (j - 1.5) * .2, [r[m] for r in R], .2, label=m)
ax.set_xticks(range(len(R))); ax.set_xticklabels(mods); ax.axhline(0, color="k", lw=.8); ax.legend(fontsize=8)
ax.set_title("保函数隐藏空间攻击下各指纹得分（victim=Qwen2.5-0.5B，72 提示）"); plt.tight_layout()
plt.savefig("/workspace/papers/figs/rotation_attack_scores.png", dpi=150)
