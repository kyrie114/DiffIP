"""由 results.json 画两张图：figs/scores_by_method.png（优点 3/4/5 三种方法并排）、
figs/layer_residual.png（优点 7 逐层残差）。"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
SA = Path(__file__).resolve().parent; FIG = SA / "figs"; FIG.mkdir(exist_ok=True)
for f in font_manager.findSystemFonts():
    if any(k in f for k in ("NotoSansCJK-Regular", "NotoSerifCJK-Regular", "wqy")):
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name(); break
plt.rcParams["axes.unicode_minus"] = False
R = json.loads((SA / "results.json").read_text())
NEGS = ["neg_smollm2_135m", "neg_smollm2_360m", "neg_gpt2", "neg_pythia_160m", "neg_qwen2_0.5b"]
LBL = {"instruct": "官方指令微调", "instruct_prune": "删4层", "drop2": "删2层", "drop6": "删6层", "drop8": "删8层",
       "drop12": "删12层", "width75": "剪宽到75%", "width50": "剪宽到50%", "width75h": "剪宽75%+恢复训练",
       "width50h": "剪宽50%+恢复训练", "vocab_ext": "扩词表+训练词嵌入", "vocab_swap": "换SmolLM2词表+训练词嵌入",
       "neg_smollm2_135m": "无关 SmolLM2-135M", "neg_smollm2_360m": "无关 SmolLM2-360M", "neg_gpt2": "无关 GPT-2",
       "neg_pythia_160m": "无关 Pythia-160M", "neg_qwen2_0.5b": "无关 Qwen2-0.5B", "instruct_wperm": "权重置换",
       "rep_perm": "表示置换", "scale": "缩放"}
SHOW = [n for n in ["instruct", "drop2", "instruct_prune", "drop6", "drop8", "drop12", "width75", "width75h", "width50",
                    "width50h", "vocab_ext", "vocab_swap"] if n in R["ours"]] + NEGS
fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
for ax, key, title in zip(axs, ["ours", "reef", "wc"], ["本方法相似度", "REEF 线性中心化核对齐", "直接权重余弦（按参数名）"]):
    vals, cols = [], []
    for n in SHOW:
        if key == "ours": v = R["ours"][n]["mean"]
        elif key == "reef": v = R["reef"][n]["cka"]
        else:
            w = R["weight_cos"].get(n); v = w["mean_cos"] if (w and w.get("status") == "ok") else None
        vals.append(v); cols.append("#c0392b" if n in NEGS else "#2e86c1")
    y = range(len(SHOW))
    ax.barh(list(y), [v if v is not None else 0 for v in vals], color=cols)
    for i, v in enumerate(vals):
        ax.text((v or 0) + 0.01, i, "无法计算" if v is None else f"{v:.3f}", va="center", fontsize=8)
    if key != "wc":
        mx = max(vals[SHOW.index(n)] for n in NEGS); ax.axvline(mx, ls="--", c="k", lw=1)
    ax.set_title(title); ax.set_xlim(0, 1.15)
axs[0].set_yticks(list(range(len(SHOW)))); axs[0].set_yticklabels([LBL.get(n, n) for n in SHOW]); axs[0].invert_yaxis()
fig.suptitle("受害模型 Qwen2.5-0.5B：衍生模型（蓝）与无关模型（红），虚线 = 无关模型最高分")
fig.tight_layout(); fig.savefig(FIG / "scores_by_method.png", dpi=200)
fig, ax = plt.subplots(figsize=(8, 4.5))
for n in [m for m in ["instruct", "instruct_prune", "drop12", "width75h", "vocab_swap", "neg_qwen2_0.5b", "neg_smollm2_360m", "neg_gpt2"] if m in R["layer_residual"]]:
    d = R["layer_residual"][n]
    ax.errorbar(range(1, 5), d["mean"], yerr=d["std"], marker="o", capsize=3, ls="--" if n in NEGS else "-", label=LBL.get(n, n))
ax.set_xticks(range(1, 5)); ax.set_xticklabels(["第1个取样层", "第2个", "第3个", "第4个"])
ax.set_ylabel("对齐后平均残差（越小越像）"); ax.legend(fontsize=8, ncol=2); ax.set_title("逐层残差：32 条提示 × 3 个种子组的均值 ± 标准差")
fig.tight_layout(); fig.savefig(FIG / "layer_residual.png", dpi=200); print("ok")
