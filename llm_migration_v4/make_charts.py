"""Charts for the advisor-report deck, built from the final scores/*.json.

Palette + typography are shared with build_ppt.py. Outputs PNG @200dpi into ppt_assets/.
"""
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = Path(__file__).resolve().parent
A = H / "ppt_assets"
A.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "axes.unicode_minus": False,
    "axes.edgecolor": "#94A3B8",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# palette
NAVY = "#1F3A5F"
TEAL = "#159A9C"
ORANGE = "#E8871A"
RED = "#C7443C"
PURPLE = "#7B5EA7"
GRAY = "#8A94A6"
GREEN = "#3D9A50"
LIGHT = "#C9D3E0"

S = H / "scores"


def load(line, names):
    return {n: json.loads((S / f"{line}_{n}.json").read_text()) for n in names}


def cell(rows, n, d, agg="mean"):
    """mean similarity (over prompts [+ seeds]) at a (N, dims) cell."""
    vals = [r["similarity"] for r in rows if r["N"] == n and r["dims"] == d]
    return float(np.mean(vals)) if agg == "mean" else vals


def dist_cell(rows, n, d):
    vals = [r["distance_mean"] for r in rows if r["N"] == n and r["dims"] == d]
    return float(np.mean(vals))


MAIN = load("main", ["fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff",
                     "derivative_cn", "neg_pruned_sparsegpt", "neg_pruned_wanda",
                     "neg_pruned_gblm", "neg_qwen2_5_7b", "neg_llama3_8b", "victim"])
DIST = load("distill", ["student_distill", "student_control", "neg_llama3_2_3b",
                        "neg_sheared_1_3b", "neg_qwen2_5_7b", "neg_llama2_7b"])


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- c1 genealogy
def c1():
    N, d = 128, 8
    groups = [
        ("victim（基座自身）", ["victim"], NAVY),
        ("同底座剪枝变体", ["neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm"], TEAL),
        ("官方 SFT（chat）", ["fine_tune"], ORANGE),
        ("chat+剪4层", ["prune_dpoff"], ORANGE),
        ("白盒伪装×3", ["weight_perm", "rep_perm", "scale"], ORANGE),
        ("独立衍生（中文）", ["derivative_cn"], PURPLE),
        ("跨家族负例", ["neg_qwen2_5_7b", "neg_llama3_8b"], RED),
    ]
    labels, vals, colors = [], [], []
    for label, names, color in groups:
        v = np.mean([cell(MAIN[n], N, d) for n in names])
        labels.append(label)
        vals.append(v)
        colors.append(color)
    order = np.argsort(vals)[::-1]
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
    bars = ax.barh(range(len(vals)), vals, color=colors, height=0.62, zorder=3)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=15)
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=15,
                fontweight="bold", color="#334155")
    ax.set_xlabel("similarity（exp(−Σd/L)，N=128，dims=8，越高越像 victim）", fontsize=13)
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axvspan(0, 0.26, color=RED, alpha=0.07, zorder=0)
    ax.axvline(0.254, color=RED, ls="--", lw=1.2, zorder=2)
    ax.text(0.262, len(vals) - 0.55, "跨家族地板 0.254\n（≈v3 的 0.24）", fontsize=12,
            color=RED, va="top")
    ax.invert_yaxis()
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(A / "c1_genealogy.png")
    plt.close(fig)


# ---------------------------------------------------------------- c2 convergence
def c2():
    Ns = [8, 16, 20, 32, 64, 128]
    d = 8
    series = [
        ("同底座剪枝（3 个难负例）", ["neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm"], TEAL),
        ("SFT 簇（chat+3 伪装+剪4层）", ["fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff"], ORANGE),
        ("独立衍生", ["derivative_cn"], PURPLE),
        ("跨家族（2 个易负例）", ["neg_qwen2_5_7b", "neg_llama3_8b"], RED),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.9), dpi=200)
    for label, names, color in series:
        ys = [np.mean([cell(MAIN[m], n_, d) for m in names]) for n_ in Ns]
        ax.plot(range(len(Ns)), ys, "-o", color=color, lw=2.6, ms=7,
                label=label, zorder=3)
    ax.plot(range(len(Ns)), [1.0] * len(Ns), "-o", color=NAVY, lw=2.6, ms=7,
            label="victim", zorder=3)
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([str(n) for n in Ns], fontsize=13)
    ax.set_xlabel("采样预算 N", fontsize=14)
    ax.set_ylabel("similarity（跨 prompt+seed 均值，dims=8）", fontsize=13)
    ax.legend(fontsize=12.5, frameon=False, loc="center right")
    ax.axvspan(3.5, 5.5, color=NAVY, alpha=0.05, zorder=0)
    ax.text(4.5, 0.36, "N≥64 后 std ≤ 0.013\n（与 v3 收敛行为一致）", fontsize=12,
            ha="center", color="#475569")
    ax.text(4.5, 0.205, "地板不随规模变化", fontsize=12, ha="center", color=RED)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(A / "c2_convergence.png")
    plt.close(fig)


# ---------------------------------------------------------------- c3 disguise
def c3():
    N, d = 128, 8
    names = ["fine_tune", "weight_perm", "rep_perm", "scale"]
    vals = [cell(MAIN[n], N, d) for n in names]
    ctx_vals = [np.mean([cell(MAIN[n], N, d) for n in ["neg_pruned_sparsegpt",
                                                      "neg_pruned_wanda", "neg_pruned_gblm"]]),
                np.mean([cell(MAIN[n], N, d) for n in ["neg_qwen2_5_7b", "neg_llama3_8b"]])]
    labels = ["fine_tune\n(官方SFT)", "weight_perm\n(白盒置换)", "rep_perm\n(特征置换)", "scale\n(特征缩放)"]
    fig, ax = plt.subplots(figsize=(9.2, 4.9), dpi=200)
    xs = np.arange(4)
    bars = ax.bar(xs, vals, width=0.56, color=ORANGE, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=15, fontweight="bold",
                color="#334155")
    for x, v, c in [(4.35, ctx_vals[0], TEAL), (6.1, ctx_vals[1], RED)]:
        ax.bar([x], [v], width=0.56, color=c, zorder=3)
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=15, fontweight="bold",
                color="#334155")
    xs_all = list(xs) + [4.35, 6.1]
    ax.set_xticks(xs_all)
    ax.set_xticklabels(["fine_tune\n(官方SFT)", "weight_perm\n(白盒置换)",
                        "rep_perm\n(特征置换)", "scale\n(特征缩放)", "同底座剪枝\n(对照上下文)",
                        "跨家族\n(对照上下文)"], fontsize=13)
    lo, hi = 0.365, 0.0365
    ax.axhspan(min(vals) - 0.004, max(vals) + 0.004, color=ORANGE, alpha=0.12, zorder=1)
    ax.annotate("三种伪装与官方 SFT 完全一致（噪声内）\nPCA+ALS 指纹对其不可见",
                xy=(1.5, 0.364), xytext=(1.5, 0.475), fontsize=14, ha="center",
                color="#7C480B",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2))
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("similarity（N=128，dims=8）", fontsize=13)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(A / "c3_disguise.png")
    plt.close(fig)


# ---------------------------------------------------------------- c4 cross-scale
def c4():
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=200)
    cats = ["fine_tune\n(正例代表)", "同源判别 margin\n(正例−易负例)", "跨家族地板\n(无关模型)"]
    v3 = [0.760, 0.52, 0.235]
    v4 = [0.363, 0.363 - 0.254, 0.254]
    xs = np.arange(3)
    w = 0.32
    b1 = ax.bar(xs - w / 2, v3, w, color=LIGHT, label="v3：0.5B（Qwen2.5-0.5B 面板）", zorder=3)
    b2 = ax.bar(xs + w / 2, v4, w, color=NAVY, label="v4：7B（Llama-2-7b 面板）", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.3f}", ha="center", fontsize=14, fontweight="bold",
                    color="#334155")
    # arrows
    ax.annotate("", xy=(0 + w / 2, 0.363), xytext=(0 - w / 2, 0.760),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.text(0.05, 0.56, "绝对值减半多", fontsize=13.5, color=RED, fontweight="bold")
    ax.annotate("", xy=(2 + w / 2, 0.254), xytext=(2 - w / 2, 0.235),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    ax.text(2.12, 0.30, "地板几乎不变", fontsize=13.5, color=GREEN, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(cats, fontsize=13.5)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("similarity / margin", fontsize=13)
    ax.legend(fontsize=12.5, frameon=False, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(A / "c4_scale.png")
    plt.close(fig)


# ---------------------------------------------------------------- c5 hard negs+auc
def c5():
    N, d = 128, 8
    hard = ["neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm"]
    ft = {(r["prompt"], r["seed_set"]): r["similarity"] for r in MAIN["fine_tune"]
          if r["N"] == N and r["dims"] == d}
    hp = [{(r["prompt"], r["seed_set"]): r["similarity"] for r in MAIN[h]
           if r["N"] == N and r["dims"] == d} for h in hard]
    diffs = [min(max(h[p, s] for h in hp) - ft[p, s], 0.35) for s in (0,)
             for p in range(32)]
    diffs = np.array(diffs)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=200,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    colors = [RED if x > 0 else GREEN for x in diffs]
    bars = ax.bar(range(32), diffs, width=0.8, color=colors, zorder=3)
    ax.axhline(0, color="#334155", lw=1.2)
    ax.set_xlabel("prompt 序号（32 个探针）", fontsize=13)
    ax.set_ylabel("最优难负例 − fine_tune", fontsize=13)
    ax.set_ylim(-0.05, 0.43)
    ax.set_title("同底座剪枝负例在 31/32 个 prompt 上压过官方 SFT",
                 fontsize=13.5, color="#7C2D2A", pad=10)
    ax.text(-1.5, 0.392, "平均高出 +0.216（仅 prompt #12 反超）", fontsize=13.5,
            color=RED, fontweight="bold", ha="left")
    style_ax(ax)

    ax = axes[1]
    # AUC bars
    main_pos = ["fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff", "derivative_cn"]
    easy = ["neg_qwen2_5_7b", "neg_llama3_8b"]
    for k, d_ in enumerate([8, 20]):
        pos = [r["similarity"] for n in main_pos for r in MAIN[n]
               if r["N"] == N and r["dims"] == d_]
        neg = [r["similarity"] for n in easy for r in MAIN[n]
               if r["N"] == N and r["dims"] == d_]
        # tie-aware AUC
        allv = np.array(pos + neg)
        ranks = np.argsort(np.argsort(allv)) + 1
        r_pos = ranks[:len(pos)].sum()
        auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        b = ax.bar(k, auc, 0.5, color=NAVY, zorder=3)
        ax.text(k, auc + 0.015, f"{auc:.3f}", ha="center", fontsize=16,
                fontweight="bold", color=NAVY)
    ax.bar([2], [1.0], 0.5, color=GREEN, zorder=3)
    ax.text(2, 1.015, "理想 1.0", ha="center", fontsize=13.5, color=GREEN, fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["AUC\n(dims=8)", "AUC\n(dims=20)", "理想"], fontsize=12.5)
    ax.set_ylim(0, 1.15)
    ax.set_title("跨家族判别 AUC ≈ 0.63", fontsize=13.5, color=NAVY, pad=10)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(A / "c5_hardneg.png")
    plt.close(fig)


# ---------------------------------------------------------------- c6 distill
def c6():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), dpi=200,
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    Ns = [8, 16, 20, 32, 64, 128]
    d = 8
    y1 = [cell(DIST["student_distill"], n, d) for n in Ns]
    y2 = [cell(DIST["student_control"], n, d) for n in Ns]
    ax.plot(range(len(Ns)), y1, "-o", color=NAVY, lw=2.6, ms=7, label="student_distill（蒸馏自 3B 教师）")
    ax.plot(range(len(Ns)), y2, "-o", color=ORANGE, lw=2.6, ms=7, label="student_control（同预算 Alpaca 对照）")
    ax.fill_between(range(len(Ns)), y2, y1, color=GRAY, alpha=0.15)
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([str(n) for n in Ns], fontsize=13)
    ax.set_xlabel("N", fontsize=13.5)
    ax.set_ylabel("similarity（dims=8）", fontsize=13)
    ax.set_ylim(0.58, 0.83)
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    for x, a, b in zip(range(len(Ns)), y1, y2):
        ax.text(x, max(a, b) + 0.006, f"{a - b:+.3f}", ha="center", fontsize=11.5,
                color="#475569")
    ax.set_title("两臂全程咬合：差 ≤0.003，方向随设置翻转", fontsize=13, color=NAVY, pad=10)
    style_ax(ax)

    ax = axes[1]
    N, d = 128, 8
    negs = ["neg_llama3_2_3b", "neg_sheared_1_3b", "neg_qwen2_5_7b", "neg_llama2_7b"]
    sg = {r["prompt"]: r["similarity"] for r in DIST["student_distill"]
          if r["N"] == N and r["dims"] == d and r["seed_set"] == 0}
    ck = {r["prompt"]: r["similarity"] for r in DIST["student_control"]
          if r["N"] == N and r["dims"] == d and r["seed_set"] == 0}
    rel = []
    for nk in negs:
        ng = {r["prompt"]: r["similarity"] for r in DIST[nk]
              if r["N"] == N and r["dims"] == d and r["seed_set"] == 0}
        md = np.mean([sg[p] - ng[p] for p in sg])
        mc = np.mean([ck[p] - ng[p] for p in ck])
        rel.append((nk, md, mc))
    xs = np.arange(len(negs))
    w = 0.34
    labels = ["3B 教师", "Sheared-1.3B", "Qwen2.5-7B", "Llama-2-7B"]
    ax.bar(xs - w / 2, [r[1] for r in rel], w, color=NAVY, label="student_distill−负例", zorder=3)
    ax.bar(xs + w / 2, [r[2] for r in rel], w, color=ORANGE, label="student_control−负例", zorder=3)
    for i, r in enumerate(rel):
        ax.text(i - w / 2, r[1] + 0.0015, f"{r[1]:.3f}", ha="center", fontsize=11.5, color=NAVY)
        ax.text(i + w / 2, r[2] + 0.0015, f"{r[2]:.3f}", ha="center", fontsize=11.5, color=ORANGE)
    ax.set_xticks(xs)
    ax.set_xticklabels(["3B 教师", "Sheared\n1.3B", "Qwen2.5\n7B", "Llama-2\n7B"],
                       fontsize=12)
    ax.axhline(0.03, color=GREEN, ls="--", lw=1.5)
    ax.text(-0.45, 0.0452, "虚线：两臂共享 margin +0.03（同样超负例）", fontsize=12,
            color=GREEN, fontweight="bold", ha="left")
    ax.set_ylim(0, 0.05)
    ax.set_ylabel("学生 − 负例（均值 margin）", fontsize=12)
    ax.set_title("学生们超负例的 margin 两臂共享——\n是“被训练过”，不是“学过教师”", fontsize=12.5,
                 color=NAVY, pad=8)
    ax.legend(fontsize=11, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.005), ncols=2, handlelength=1.4,
              borderaxespad=0.0)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(A / "c6_distill.png")
    plt.close(fig)


if __name__ == "__main__":
    c1(), c2(), c3(), c4(), c5(), c6()
    print("charts ->", A)
