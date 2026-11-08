"""Charts for the v2 advisor deck (16:9, large type, white cards).

All numbers are read from the committed result JSONs / score tables under
llm_migration_v4/.  Output: 汇报PPT_v2/figs/*.png @200dpi.

Run:
    cd llm_migration_v4
    /home/lushunxin/miniconda3/envs/diffip-llm/bin/python 汇报PPT_v2/make_charts_v2.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

H = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

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
    "font.size": 12,
})

NAVY = "#1F3A5F"
TEAL = "#159A9C"
ORANGE = "#E8871A"
RED = "#C7443C"
PURPLE = "#7B5EA7"
GRAY = "#8A94A6"
GREEN = "#3D9A50"
LIGHT = "#C9D3E0"
INK = "#1E293B"
MUT = "#64748B"

STRONG = ["neg_pruned_sparsegpt", "neg_pruned_wanda", "neg_pruned_gblm"]
SFT = ["fine_tune", "weight_perm", "rep_perm", "scale", "prune_dpoff"]
NEG = ["neg_qwen2_5_7b", "neg_llama3_8b"]
SHORT = {
    "neg_qwen2_5_7b": "Qwen2.5-7B",
    "neg_llama3_8b": "Llama-3-8B",
    "neg_llama1_7b": "llama-7b-hf",
    "neg_baichuan_7b": "Baichuan-7B",
    "neg_mpt_7b": "MPT-7B",
}


# ------------------------------------------------------------------ helpers
def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name, rect=(0, 0.055, 1, 1)):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("saved", OUT / name)


def load_rows(subdir, name):
    rows = json.loads((H / subdir / f"main_{name}.json").read_text())
    return [r for r in rows if r["N"] == 128 and r["dims"] == 8]


def cell_stats(subdir, name):
    rows = load_rows(subdir, name)
    per = [np.mean([r["similarity"] for r in rows if r["seed_set"] == s])
           for s in (0, 1, 2)]
    return float(np.mean(per)), float(np.std(per))


def grid(rows):
    return {(r["prompt"], r["seed_set"]): r["similarity"] for r in rows}


def roc_points(pos, neg):
    ths = np.unique(np.r_[pos, neg])[::-1]
    fpr = np.array([np.mean(neg >= t) for t in ths])
    tpr = np.array([np.mean(pos >= t) for t in ths])
    return fpr, tpr


# ------------------------------------------------------------------ 1 baseline
def fig_baseline():
    groups = [
        ("victim（基座自身）", ["victim"], NAVY),
        ("同底座稀疏剪枝 ×3", STRONG, TEAL),
        ("SFT 簇 ×5", SFT, ORANGE),
        ("独立衍生（中文）", ["derivative_cn"], PURPLE),
        ("跨家族负例 ×2", NEG, RED),
    ]
    means, dots, ranges, colors = [], [], [], []
    for _, names, c in groups:
        vals = [cell_stats("scores", n)[0] for n in names]
        means.append(float(np.mean(vals)))
        dots.append(vals)
        ranges.append((min(vals), max(vals)))
        colors.append(c)

    fig, ax = plt.subplots(figsize=(10.8, 5.7), dpi=200)
    ys = np.arange(len(groups))
    for i, (m, c) in enumerate(zip(means, colors)):
        ax.barh(i, m, height=0.46, color=c, alpha=0.22, zorder=2)
        ax.plot([ranges[i][0], ranges[i][1]], [i, i], color=c, lw=2.4,
                solid_capstyle="round", zorder=3)
        ax.scatter(dots[i], [i] * len(dots[i]), s=62, color=c,
                   edgecolor="white", linewidth=0.8, zorder=4)
        label_x = max(ranges[i][1], m) + 0.030
        ax.text(label_x, i, f"{m:.3f}", va="center", fontsize=15,
                fontweight="bold", color=c, zorder=5)

    ax.axvspan(0.234, 0.274, color=RED, alpha=0.07, zorder=0)
    ax.axvline(0.254, color=RED, ls="--", lw=1.4, zorder=1)
    ax.text(0.258, 4.55, "跨家族地板 0.254", color=RED, fontsize=11.5,
            va="center")
    ax.annotate("", xy=(0.254, 2.52), xytext=(0.3627, 2.52),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.2))
    ax.text(0.308, 2.62, "SFT→地板 +0.109", ha="center", fontsize=11.5,
            color=NAVY, bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.9, pad=1.5))
    ax.annotate("", xy=(0.3627, 1.52), xytext=(0.5415, 1.52),
                arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.2))
    ax.text(0.452, 1.62, "结构变体→SFT +0.179", ha="center", fontsize=11.5,
            color=TEAL, bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.9, pad=1.5))

    ax.set_yticks(ys)
    ax.set_yticklabels([g[0] for g in groups], fontsize=13.5)
    ax.set_xlim(0, 1.10)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("similarity（越高越像 victim；N=128，8 维，均匀 4 层）", fontsize=12.5)
    ax.set_ylim(4.7, -0.55)
    fig.suptitle("谱系排序清楚：结构变体 0.542 > SFT 0.363 > 独立衍生 0.311 > 跨家族 0.254",
                 fontsize=15.2, color=NAVY, fontweight="bold", y=0.985)
    style_ax(ax)
    fig.text(0.01, 0.008,
             "点 = 单个模型；横线 = 组内范围；多数 3-seed std 小于点大小。"
             "SFT 簇含 fine_tune / weight_perm / rep_perm / scale / prune_dpoff。",
             fontsize=9.5, color=MUT)
    save(fig, "fig_baseline.png", rect=(0, 0.065, 1, 0.93))


# ------------------------------------------------------------------ 2 distill
def fig_distill():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), dpi=200,
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    Ns = [8, 16, 20, 32, 64, 128]
    y1, y2, e1, e2 = [], [], [], []
    for N in Ns:
        for name, ys, es in (("student_distill", y1, e1),
                             ("student_control", y2, e2)):
            rows = json.loads((H / "scores" / f"distill_{name}.json").read_text())
            rs = [r for r in rows if r["N"] == N and r["dims"] == 8]
            per = [np.mean([r["similarity"] for r in rs if r["seed_set"] == s])
                   for s in (0, 1, 2)]
            ys.append(float(np.mean(per)))
            es.append(float(np.std(per)))
    x = np.arange(len(Ns))
    ax.errorbar(x, y1, yerr=e1, fmt="-o", color=NAVY, lw=2.6, ms=7,
                capsize=3, label="student_distill（蒸馏自 3B 教师）", zorder=3)
    ax.errorbar(x, y2, yerr=e2, fmt="-o", color=ORANGE, lw=2.6, ms=7,
                capsize=3, label="student_control（同预算 Alpaca 对照）", zorder=3)
    ax.axhspan(0.5897, 0.5908, color=GRAY, alpha=0.25, zorder=0)
    ax.text(0.1, 0.5932, "教师 / 4 个负例：0.590–0.591", fontsize=11,
            color="#475569")
    ax.annotate("-0.003", xy=(5, 0.6217), xytext=(4.2, 0.650),
                fontsize=13, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in Ns], fontsize=12.5)
    ax.set_xlabel("采样预算 N", fontsize=12.5)
    ax.set_ylabel("similarity（dims=8）", fontsize=12.5)
    ax.set_ylim(0.575, 0.835)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.set_title("两条学习曲线全程咬合", fontsize=13.5, color=NAVY, pad=9)
    style_ax(ax)

    ax = axes[1]
    labels = ["student\ndistill", "student\ncontrol", "3B\n教师", "4 个负例\n均值"]
    vals = [y1[-1], y2[-1], 0.5908, 0.5903]
    errs = [e1[-1], e2[-1], 0.0025, 0.002]
    cols = [NAVY, ORANGE, GRAY, RED]
    bars = ax.bar(range(4), vals, width=0.58, color=cols, alpha=0.88, zorder=3)
    ax.errorbar(range(4), vals, yerr=errs, fmt="none", ecolor="#334155",
                capsize=4, lw=1.2, zorder=4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=12.5,
                fontweight="bold", color="#334155")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylim(0.55, 0.66)
    ax.set_ylabel("N=128 的 similarity", fontsize=12)
    ax.text(0.5, 0.657,
            "distill − control = −0.003\nAUC 0.493\n96 对中 40 对 distill 更高",
            ha="center", va="top", fontsize=10.5, color=RED,
            fontweight="bold", linespacing=1.15)
    ax.set_title("学生与对照差在噪声内", fontsize=13.5, color=NAVY, pad=9)
    style_ax(ax)
    fig.suptitle("蒸馏线：教师与负例都在 0.590–0.591，学生高约 0.03；两臂共享这个“被训练过”的 margin",
                 fontsize=14.2, color=NAVY, fontweight="bold", y=0.985)
    save(fig, "fig_distill.png", rect=(0, 0.02, 1, 0.92))


# ------------------------------------------------------------------ 3 mid2
def fig_mid2():
    cats = ["结构变体\n（3 个稀疏剪枝）", "SFT 簇\n（5 个）",
            "独立衍生", "跨家族\n（2 个）"]
    even = [np.mean([cell_stats("scores", n)[0] for n in STRONG]),
            np.mean([cell_stats("scores", n)[0] for n in SFT]),
            cell_stats("scores", "derivative_cn")[0],
            np.mean([cell_stats("scores", n)[0] for n in NEG])]
    mid = [np.mean([cell_stats("scores_mid2", n)[0] for n in STRONG]),
           np.mean([cell_stats("scores_mid2", n)[0] for n in SFT]),
           cell_stats("scores_mid2", "derivative_cn")[0],
           np.mean([cell_stats("scores_mid2", n)[0] for n in NEG])]
    fig, ax = plt.subplots(figsize=(10.6, 5.4), dpi=200)
    ys = np.arange(len(cats))
    for i in ys:
        ax.plot([mid[i], even[i]], [i, i], color="#CBD5E1", lw=4,
                solid_capstyle="round", zorder=1)
        ax.scatter([even[i]], [i], s=130, color=NAVY, zorder=3,
                   edgecolor="white", linewidth=1.2, label="even4（默认）" if i == 0 else None)
        ax.scatter([mid[i]], [i], s=130, color=ORANGE, zorder=3,
                   edgecolor="white", linewidth=1.2, label="mid2（消融）" if i == 0 else None)
        ax.text(even[i] + 0.012, i - 0.14, f"{even[i]:.3f}", fontsize=12,
                color=NAVY, fontweight="bold", va="bottom")
        ax.text(mid[i] + 0.012, i + 0.14, f"{mid[i]:.3f}", fontsize=12,
                color=ORANGE, fontweight="bold", va="top")
    ax.text(0.40, 2.45, "SFT→地板\n0.109 → 0.079", fontsize=12,
            color=RED, fontweight="bold", ha="left", va="center",
            bbox=dict(facecolor="#FBEFEF", edgecolor=RED, alpha=0.85,
                      boxstyle="round,pad=0.35"))
    ax.set_yticks(ys)
    ax.set_yticklabels(cats, fontsize=12.5)
    ax.set_xlim(0, 0.62)
    ax.set_ylim(3.55, -0.55)
    ax.set_xlabel("similarity（N=128，8 维）", fontsize=12.5)
    ax.legend(fontsize=11.5, frameon=False, loc="lower right")
    fig.suptitle("mid2：排序保持，但所有分数整体下移，SFT 到地板的间隔收窄",
                 fontsize=15, color=NAVY, fontweight="bold", y=0.985)
    style_ax(ax)
    fig.text(0.01, 0.008,
             "mid2 = 已存 4 个均匀层中的第 1、2 层，不是 v3 全层中间 2 层。"
             "三种线性伪装仍与 fine_tune 噪声内一致。",
             fontsize=9.5, color=MUT)
    save(fig, "fig_mid2.png", rect=(0, 0.065, 1, 0.93))


# ------------------------------------------------------------------ 4 absolute
def fig_absolute():
    pos_names = STRONG + SFT
    pos = [r["similarity"] for n in pos_names for r in load_rows("scores", n)]
    neg = [r["similarity"] for n in NEG for r in load_rows("scores", n)]
    fpr, tpr = roc_points(pos, neg)
    p = np.asarray(pos)[:, None]
    q = np.asarray(neg)[None, :]
    auc = float(((p > q).sum() + 0.5 * (p == q).sum()) / (len(pos) * len(neg)))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), dpi=200,
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.plot(fpr, tpr, color=NAVY, lw=2.6, zorder=3, label=f"部署正类 vs 跨家族（AUC {auc:.3f}）")
    ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1.2, zorder=1)
    for t, c, lab, tx, ty in [
            (0.786037998868464, RED, "t=0.786\nSFT 召回 0.06", 0.52, 0.16),
            (0.499, ORANGE, "t=0.499\nSFT 32% / 负例 23%", 0.34, 0.80)]:
        fp = np.mean(np.asarray(neg) >= t)
        tp = np.mean(np.asarray(pos) >= t)
        ax.scatter([fp], [tp], s=90, color=c, zorder=5, edgecolor="white")
        ax.annotate(lab, xy=(fp, tp), xytext=(tx, ty),
                    fontsize=11, color=c, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                              pad=1.8),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.1))
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR（跨家族负例）", fontsize=12.5)
    ax.set_ylabel("TPR（部署正类）", fontsize=12.5)
    ax.legend(fontsize=10.5, frameon=False, loc="lower right")
    ax.set_title("绝对 similarity 的 ROC：AUC 0.702", fontsize=13.5, color=NAVY, pad=9)
    style_ax(ax)

    ax = axes[1]
    groups = ["SFT 簇", "结构变体", "独立衍生", "跨家族\n负例"]
    vals = [0.3208, 0.5764, 0.2500, 0.2240]
    cols = [ORANGE, TEAL, PURPLE, RED]
    ax.bar(range(4), vals, width=0.58, color=cols, alpha=0.9, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.018, f"{v:.0%}", ha="center", fontsize=13,
                fontweight="bold", color="#334155")
    ax.axhline(0.90, color=GREEN, ls="--", lw=1.3)
    ax.text(-0.48, 0.925, "正例目标 ≥90%", ha="left", fontsize=10.5,
            color=GREEN, bbox=dict(facecolor="white", edgecolor="none",
                                   alpha=0.85, pad=1.5))
    ax.axhline(0.10, color=RED, ls="--", lw=1.3)
    ax.text(-0.48, 0.125, "负例目标 ≤10%", ha="left", fontsize=10.5,
            color=RED, bbox=dict(facecolor="white", edgecolor="none",
                                 alpha=0.85, pad=1.5))
    ax.set_xticks(range(4))
    ax.set_xticklabels(groups, fontsize=11.5)
    ax.set_xlim(-0.6, 3.55)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("阈值 t=0.499 下的接受比例", fontsize=12)
    ax.set_title("把 0.5B 阈值搬到 7B：既漏正例，又误接负例",
                 fontsize=13.5, color=NAVY, pad=9)
    style_ax(ax)
    fig.suptitle("绝对分数：逐 prompt std ≈0.25，fine_tune 与负例的逐 prompt 均值相关 0.956——全局阈值对不上",
                 fontsize=14.0, color=NAVY, fontweight="bold", y=0.985)
    save(fig, "fig_absolute.png", rect=(0, 0.02, 1, 0.92))


# ------------------------------------------------------------------ 5 relative
def fig_relative():
    pairs = [("neg_qwen2_5_7b", "neg_llama3_8b", 0.016085863066372663,
              "Qwen2.5-7B → Llama-3-8B", "1.00 / 0.97", "0.00 / 0.00"),
             ("neg_llama3_8b", "neg_qwen2_5_7b", 0.013566225875955462,
              "Llama-3-8B → Qwen2.5-7B", "1.00 / 0.975", "0.031 / 0.00")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.1), dpi=200, sharey=True)
    rng = np.random.default_rng(7)
    for ax, (anchor, test, t, title, sft_rec, fpr_txt) in zip(axes, pairs):
        ga, gt, gf = grid(load_rows("scores", anchor)), grid(load_rows("scores", test)), grid(load_rows("scores", "fine_tune"))
        fm = np.array([gf[k] - ga[k] for k in gf])
        nm = np.array([gt[k] - ga[k] for k in gt])
        bp = ax.boxplot([fm, nm], positions=[0, 1], widths=0.48,
                        patch_artist=True, showfliers=False, zorder=2)
        for patch, c in zip(bp["boxes"], [ORANGE, RED]):
            patch.set_facecolor(c)
            patch.set_alpha(0.18)
            patch.set_edgecolor(c)
        for med in bp["medians"]:
            med.set_color(NAVY)
            med.set_linewidth(2)
        for i, (vals, c) in enumerate([(fm, ORANGE), (nm, RED)]):
            xj = i + rng.uniform(-0.13, 0.13, len(vals))
            ax.scatter(xj, vals, s=16, color=c, alpha=0.35, zorder=3,
                       edgecolor="none")
        ax.axhline(t, color=NAVY, ls="--", lw=1.5, zorder=4)
        ax.text(1.42, t + 0.004, f"阈值 {t:.4f}", ha="right", fontsize=10.5,
                color=NAVY)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["SFT\n(fine_tune)", "留出负例"], fontsize=11.5)
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(-0.03, 0.25)
        ax.set_title(title, fontsize=13.2, color=NAVY, pad=9)
        ax.text(0.03, 0.96,
                f"留出 SFT 召回 {sft_rec}\n留出 FPR {fpr_txt}\nAUC 0.9998",
                transform=ax.transAxes, fontsize=10.5, color="#334155",
                va="top")
        ax.text(0.03, 0.04, "SFT 最大 0.506 未显示", transform=ax.transAxes,
                fontsize=9, color=MUT)
        style_ax(ax)
    axes[0].set_ylabel("m = sim(模型) − sim(同 prompt 参照)", fontsize=12)
    fig.suptitle("相对分数：同一条 prompt 上减参照后，SFT 与留出负例完全分开",
                 fontsize=15, color=NAVY, fontweight="bold", y=0.985)
    fig.text(0.01, 0.008,
             "每点 = 一个 (prompt, seed) 格子；箱体为四分位。阈值由 seed 0 的 32 个负例最大值定，"
             "seed 1/2 留出。", fontsize=9.5, color=MUT)
    save(fig, "fig_relative.png", rect=(0, 0.065, 1, 0.93))


# ------------------------------------------------------------------ 6 break
def fig_break():
    d = json.loads((H / "scores_break" / "summary.json").read_text())
    order = ["ident", "shuffle", "rankgauss", "mix_llama_025", "mix_llama_050",
             "mix_llama_075", "mix_deriv_050"]
    label = {
        "ident": "恒等对照",
        "shuffle": "打乱配对",
        "rankgauss": "逐维正态化",
        "mix_llama_025": "掺 Llama-3 25%",
        "mix_llama_050": "掺 Llama-3 50%",
        "mix_llama_075": "掺 Llama-3 75%",
        "mix_deriv_050": "掺中文衍生 50%",
    }
    v = {x["name"]: x for x in d["variants"]}
    fig, ax = plt.subplots(figsize=(10.8, 5.3), dpi=200)
    ys = np.arange(len(order))
    for i, name in enumerate(order):
        x = v[name]
        sim, std, wins = x["sim_n128_d8"], x["seed_std"], x["wins"]
        if name == "shuffle":
            c = RED
        elif name == "ident":
            c = NAVY
        elif name == "rankgauss":
            c = TEAL
        else:
            c = ORANGE
        ax.errorbar(sim, i, xerr=std, fmt="o", ms=10, color=c, ecolor=c,
                    capsize=4, lw=1.8, zorder=3)
        ax.text(sim + std + 0.007, i, f"{sim:.3f} ± {std:.3f}\n{wins}",
                va="center", fontsize=10.5, color=c, fontweight="bold",
                linespacing=1.15,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                          pad=1.2))
    ax.axvspan(0.234, 0.274, color=RED, alpha=0.07, zorder=0)
    ax.axvline(0.254, color=RED, ls="--", lw=1.3, zorder=1)
    ax.axvline(0.3634, color=NAVY, ls=":", lw=1.3, zorder=1)
    ax.text(0.257, -0.55, "地板 0.254", color=RED, fontsize=10.5,
            va="center", bbox=dict(facecolor="white", edgecolor="none",
                                   alpha=0.85, pad=1.5))
    ax.text(0.367, -0.55, "fine_tune 0.363", color=NAVY, fontsize=10.5,
            va="center", bbox=dict(facecolor="white", edgecolor="none",
                                   alpha=0.85, pad=1.5))
    ax.set_yticks(ys)
    ax.set_yticklabels([label[n] for n in order], fontsize=12.5)
    ax.set_xlim(0.22, 0.45)
    ax.set_ylim(len(order) - 0.5, -0.85)
    ax.set_xlabel("similarity（N=128，8 维）", fontsize=12.5)
    ax.set_title("只有打乱配对能打掉谱系；非线性与掺入只是把分数往下拉",
                 fontsize=15, color=NAVY, pad=12, fontweight="bold")
    style_ax(ax)
    fig.text(0.01, 0.008,
             "掺 Llama-3 时裁判只用 Qwen，避免把掺进去的模型当负例；"
             "胜负为 seed 0 对两个跨家族负例的逐 prompt 胜负。",
             fontsize=9.5, color=MUT)
    save(fig, "fig_break.png", rect=(0, 0.065, 1, 1))


# ------------------------------------------------------------------ 7 nonlinear
def fig_nonlinear():
    d = json.loads((H / "scores_nl" / "summary.json").read_text())
    order = ["ident", "f1_pow05", "f1_perm_pow05", "f1_tanh", "f1_scale_tanh",
             "f1_pow2", "f2_block64_pow05", "f2_lowrank", "f3_layerpow",
             "f3_layermix"]
    label = {
        "ident": "恒等对照",
        "f1_pow05": "逐维 √|x|",
        "f1_perm_pow05": "置换 + √|x|",
        "f1_tanh": "tanh",
        "f1_scale_tanh": "缩放 + tanh",
        "f1_pow2": "符号保持平方",
        "f2_block64_pow05": "块旋转 + √|x|",
        "f2_lowrank": "秩 16 tanh 残差",
        "f3_layerpow": "四层异幂",
        "f3_layermix": "四层异变换",
    }
    v = {x["name"]: x for x in d["variants"]}
    fig, ax = plt.subplots(figsize=(10.8, 5.5), dpi=200)
    ys = np.arange(len(order))
    for i, name in enumerate(order):
        x = v[name]
        sim, std, wins = x["sim_n128_d8"], x["seed_std"], x["wins_vs_cross_family"]
        c = NAVY if name == "ident" else (ORANGE if name == "f1_pow2" else TEAL)
        ax.errorbar(sim, i, xerr=std, fmt="o", ms=10, color=c, ecolor=c,
                    capsize=4, lw=1.8, zorder=3)
        ax.text(sim + std + 0.006, i, f"{sim:.3f} ± {std:.3f}\n{wins}",
                va="center", fontsize=10.5, color=c, fontweight="bold",
                linespacing=1.15,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                          pad=1.2))
    ax.axvspan(0.3454, 0.3814, color=ORANGE, alpha=0.08, zorder=0)
    ax.axvspan(0.234, 0.274, color=RED, alpha=0.07, zorder=0)
    ax.axvline(0.3634, color=NAVY, ls=":", lw=1.3, zorder=1)
    ax.axvline(0.254, color=RED, ls="--", lw=1.3, zorder=1)
    ax.text(0.347, -0.55, "稳健带 0.363±0.018", color=ORANGE, fontsize=10.5,
            va="center", bbox=dict(facecolor="white", edgecolor="none",
                                   alpha=0.85, pad=1.5))
    ax.text(0.257, -0.55, "地板 0.254", color=RED, fontsize=10.5,
            va="center", bbox=dict(facecolor="white", edgecolor="none",
                                   alpha=0.85, pad=1.5))
    ax.set_yticks(ys)
    ax.set_yticklabels([label[n] for n in order], fontsize=12)
    ax.set_xlim(0.23, 0.46)
    ax.set_ylim(len(order) - 0.5, -0.85)
    ax.set_xlabel("similarity（N=128，8 维；误差棒为 3-seed std）", fontsize=12.5)
    ax.set_title("9 种非线性伪装：8 种仍在噪声带内，最强只把分数打薄 0.019",
                 fontsize=15, color=NAVY, pad=12, fontweight="bold")
    style_ax(ax)
    fig.text(0.01, 0.008,
             "预注册判定：与 0.363 差 ≤0.018 且胜负 ≥30/32 为稳健；"
             "落到地板 0.254 的 0.020 内或胜负 ≤20/32 为攻击成功。",
             fontsize=9.5, color=MUT)
    save(fig, "fig_nonlinear.png", rect=(0, 0.065, 1, 1))


# ------------------------------------------------------------------ 8 dim
def fig_dim():
    d = json.loads((H / "exp_dim_result.json").read_text())
    rec = {r["dims"]: r for r in d["records"]}
    dims = d["dims_compared"]
    base = rec[8]["margin"]
    gain = [rec[k]["margin"] - base for k in dims]
    std = [rec[k]["std"] for k in dims]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), dpi=200,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    cols = [NAVY if k == 8 else TEAL for k in dims]
    ax.bar(range(len(dims)), gain, width=0.58, color=cols, alpha=0.9, zorder=3)
    for i, g in enumerate(gain):
        ax.text(i, g + 0.0008, f"{g:+.4f}", ha="center", fontsize=10.5,
                color="#334155", fontweight="bold")
    ax.axhline(0.02, color=RED, ls="--", lw=1.4)
    ax.text(len(dims) - 0.5, 0.0206, "替换线：8 维 +0.02", ha="right",
            fontsize=10.5, color=RED, fontweight="bold")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(k) for k in dims], fontsize=12)
    ax.set_xlabel("维数 d", fontsize=12.5)
    ax.set_ylabel("间隔相对 8 维的增益", fontsize=12)
    ax.set_ylim(0, 0.023)
    ax.set_title("6/6 通过；最大增益 +0.006，不到替换线", fontsize=13.2,
                 color=NAVY, pad=9)
    style_ax(ax)

    ax = axes[1]
    ax.plot(range(len(dims)), std, "-o", color=PURPLE, lw=2.6, ms=8, zorder=3)
    for i, s in enumerate(std):
        ax.text(i, s + 0.0006, f"{s:.4f}", ha="center", fontsize=9.5,
                color=PURPLE)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(k) for k in dims], fontsize=12)
    ax.set_xlabel("维数 d", fontsize=12.5)
    ax.set_ylabel("3-seed std", fontsize=12)
    ax.set_ylim(0.074, 0.082)
    ax.set_title("标准差 0.080 → 0.077，只降 0.003", fontsize=13.2,
                 color=NAVY, pad=9)
    style_ax(ax)
    fig.suptitle("维数：六个维全部通过，但收益 +0.006 < 替换线 +0.02，默认保留 8 维",
                 fontsize=15, color=NAVY, fontweight="bold", y=0.985)
    fig.text(0.01, 0.008,
             "间隔 = 两个参照方向里更差的 fine_tune 留出中位数；"
             "20 维有 8 个格子和已存分数差 >1e-3（最大 4.3e-3，随机 PCA 格子差）。",
             fontsize=9.5, color=MUT)
    save(fig, "fig_dim.png", rect=(0, 0.065, 1, 0.93))


# ------------------------------------------------------------------ 9 layers
def fig_layers():
    d = json.loads((H / "exp_layers_result.json").read_text())
    rec = {r["subset"]: r for r in d["records"]}
    order = ["control", "uniform8", "all", "last4"]
    label = {"control": "均匀 4 层\n(even4)", "uniform8": "均匀 8 层",
             "all": "全部 32 层", "last4": "最后 4 层"}
    gap = [rec[k]["margin"] for k in order]
    passed = [rec[k]["passed"] for k in order]
    fig, ax = plt.subplots(figsize=(10.4, 5.1), dpi=200)
    xs = np.arange(len(order))
    cols = [GREEN if p else RED for p in passed]
    ax.bar(xs, gap, width=0.56, color=cols, alpha=0.88, zorder=3)
    for i, (g, p) in enumerate(zip(gap, passed)):
        ax.text(i, g + 0.0025, f"{g:.3f}", ha="center", fontsize=13,
                fontweight="bold", color="#334155")
        ax.text(i, g / 2, "通过" if p else "不通过", ha="center", va="center",
                fontsize=13, color="white", fontweight="bold")
    ax.axhline(rec["control"]["margin"], color=NAVY, ls="--", lw=1.3)
    ax.text(len(order) - 0.55, rec["control"]["margin"] + 0.002,
            "even4 峰值 0.092", ha="right", fontsize=10.5, color=NAVY)
    ax.annotate("SFT 召回 0.71 / 0.79\n一个方向 FPR 0.125",
                xy=(1.94, 0.047), xytext=(1.55, 0.083), fontsize=10.5,
                color=RED, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.annotate("SFT 召回 0.66–0.78", xy=(2.97, 0.037), xytext=(2.92, 0.066),
                fontsize=10.5, color=RED, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.set_xticks(xs)
    ax.set_xticklabels([label[k] for k in order], fontsize=12)
    ax.set_ylim(0, 0.108)
    ax.set_ylabel("间隔（fine_tune 留出中位数）", fontsize=12)
    ax.set_title("层数：均匀 4 层是峰值，加层反而变差", fontsize=15,
                 color=NAVY, pad=12, fontweight="bold")
    style_ax(ax)
    fig.text(0.01, 0.008,
             "小盘重采全部 32 层：victim / fine_tune / SparseGPT / Qwen2.5-7B / Llama-3-8B；"
             "Qwen 28 层，其余 32 层。通过线：召回 ≥0.90、FPR ≤0.10。",
             fontsize=9.5, color=MUT)
    save(fig, "fig_layers.png", rect=(0, 0.065, 1, 1))


# ------------------------------------------------------------------ 10 unrel matrix
def fig_unrel_matrix():
    d = json.loads((H / "exp_unrel_result.json").read_text())
    models = ["neg_qwen2_5_7b", "neg_llama3_8b", "neg_llama1_7b",
              "neg_baichuan_7b", "neg_mpt_7b"]
    cells = {(x["anchor"], x["test"]): x for x in d["directions"]}
    fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=200)
    for i, a in enumerate(models):
        for j, t in enumerate(models):
            if a == t:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="#E2E8F0", edgecolor="white",
                                       lw=1.5, zorder=2))
                ax.text(j, i, "—", ha="center", va="center", fontsize=13,
                        color=MUT, zorder=3)
                continue
            x = cells[(a, t)]
            ok = x["pass"]
            f1, f2 = x["held_fpr"]
            face = "#E8F5E9" if ok else "#FBEFEF"
            edge = GREEN if ok else RED
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                   edgecolor=edge, lw=2.0 if not ok else 1.0,
                                   zorder=2))
            mark = "✓" if ok else "×"
            ax.text(j, i - 0.12, mark, ha="center", va="center", fontsize=15,
                    color=edge, fontweight="bold", zorder=3)
            ax.text(j, i + 0.20, f"{f1:.3f} / {f2:.3f}", ha="center",
                    va="center", fontsize=9.5, color="#334155", zorder=3)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([SHORT[m] for m in models], fontsize=11.5)
    ax.set_yticklabels([SHORT[m] for m in models], fontsize=11.5)
    ax.set_xlabel("留出负例（test）", fontsize=12.5)
    ax.set_ylabel("参照（anchor）", fontsize=12.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("20 对参照/留出：18 对通过，2 对失败都在 MPT 当留出负例",
                 fontsize=14.5, color=NAVY, pad=12, fontweight="bold")
    fig.text(0.01, 0.008,
             "单元格 = 留出 seed 1 / seed 2 的 FPR；✓ = 两 seed 都过（强正例与 SFT 召回 ≥0.90，FPR ≤0.10），"
             "× = 未通过。对角线不计算。",
             fontsize=9.5, color=MUT)
    save(fig, "fig_unrel_matrix.png", rect=(0, 0.04, 1, 1))


# ------------------------------------------------------------------ 11 unrel diagnosis
def fig_unrel_diag():
    pairs = [("neg_qwen2_5_7b", "neg_mpt_7b", 0.007383735078996548,
              "Qwen2.5-7B → MPT-7B", "4/32", "0.125", "+0.0004"),
             ("neg_llama3_8b", "neg_mpt_7b", 0.003914897781212956,
              "Llama-3-8B → MPT-7B", "6/32", "0.188", "+0.0018")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), dpi=200, sharey=True)
    for ax, (anchor, test, t, title, above, fpr, shift) in zip(axes, pairs):
        ga = grid([r for r in json.loads((H / "scores_dim" / f"main_{anchor}.json").read_text())
                   if r["N"] == 128 and r["dims"] == 8])
        gt = grid([r for r in json.loads((H / "scores_unrel" / f"main_{test}.json").read_text())
                   if r["N"] == 128 and r["dims"] == 8])
        for k, c, lab in [(0, NAVY, "seed 0（标定）"), (1, ORANGE, "seed 1（留出）"),
                          (2, RED, "seed 2（留出）")]:
            m = np.array([gt[(p, k)] - ga[(p, k)] for p in range(32)])
            m = np.sort(m)[::-1]
            ax.plot(range(1, 33), m, "-o", ms=4.5, lw=1.6, color=c, label=lab,
                    zorder=3)
        ax.axhline(t, color=NAVY, ls="--", lw=1.5, zorder=2,
                   label="标定阈值 t（seed 0 最大负例）")
        ax.axhline(t + 0.002, color=GREEN, ls=":", lw=1.6, zorder=2,
                   label="t + 0.002（事后保护带）")
        ax.axvline(3.5, color=GRAY, ls=":", lw=1.1, zorder=1)
        ax.text(3.7, 0.0185, "允许 3/32", fontsize=9.5, color=GRAY,
                va="top", bbox=dict(facecolor="white", edgecolor="none",
                                    alpha=0.85, pad=1.5))
        ax.text(31.8, t - 0.0042, f"t = {t:.4f}", ha="right", fontsize=10.5,
                color=NAVY)
        ax.text(0.03, 0.04,
                f"seed 2：{above} 超阈（FPR {fpr}）\n阈值需抬高 {shift}",
                transform=ax.transAxes, fontsize=10.5, color=RED,
                va="bottom", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          pad=2.0))
        ax.set_xlim(0, 33)
        ax.set_ylim(-0.013, 0.020)
        ax.set_xlabel("32 条 prompt 按 margin 降序", fontsize=11.5)
        ax.set_title(title, fontsize=13.2, color=NAVY, pad=9)
        ax.legend(fontsize=8.6, frameon=False, loc="upper right",
                  handlelength=1.8, labelspacing=0.3)
        style_ax(ax)
    axes[0].set_ylabel("margin = sim(MPT) − sim(参照)", fontsize=12)
    fig.suptitle("失败在阈值尾部：排序没坏（AUC 0.9999），只需 +0.0004 / +0.0018",
                 fontsize=14.8, color=NAVY, fontweight="bold", y=0.985)
    fig.text(0.01, 0.008,
             "阈值 = seed 0 上 32 个负例 margin 的最大值。事后加 +0.002 保护带可 20/20，"
             "但这是事后规则，不计入预注册判定。", fontsize=9.5, color=MUT)
    save(fig, "fig_unrel_diag.png", rect=(0, 0.065, 1, 0.93))


if __name__ == "__main__":
    fig_baseline()
    fig_distill()
    fig_mid2()
    fig_absolute()
    fig_relative()
    fig_break()
    fig_nonlinear()
    fig_dim()
    fig_layers()
    fig_unrel_matrix()
    fig_unrel_diag()
    print("all charts ->", OUT)
