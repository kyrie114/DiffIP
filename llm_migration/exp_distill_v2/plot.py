import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
R = [r for r in json.load(open("results.json"))["rows"] if r["subset"] == "all72"]; S = json.load(open("strength.json"))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ms = ["CKA_std", "CKA_plain", "CKA_plain_7avg", "DiffIP20"]; w = 0.11
for k, r in enumerate(R):
    ax[0].bar(np.arange(4) + (k - 3) * w, [r[m] for m in ms], w, label=r["model"], color="tab:red" if r["label"] else None, edgecolor="k" if r["label"] else "none")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(ms); ax[0].set_ylim(0.78, 0.97); ax[0].legend(fontsize=7, ncol=2); ax[0].set_title("与教师 Qwen2.5-0.5B-Instruct 的相似度（72 提示；红=蒸馏学生）")
names = [s["model"] for s in S[:3]]; ks = ["teacher_nll_per_byte", "teacher_top1", "rougeL_vs_teacher"]
for j, kk in enumerate(ks): ax[1].bar(np.arange(3) + (j - 1) * 0.27, [s[kk] for s in S[:3]], 0.27, label=kk)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names); ax[1].legend(fontsize=8); ax[1].set_title("蒸馏强度（120 条留出指令）")
plt.tight_layout(); plt.savefig("distill_v2.png", dpi=140)
