# -*- coding: utf-8 -*-
"""论文级方法/原理图（300 dpi，供汇报 PPT 使用）。
fig1: 方法全流程（配对扰动采样 → 特征提取 → ALS 评分）
fig2: ALS 对齐原理（同源 vs 异源特征云）
fig3: 两线谱系面板
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"
PURPLE, GREY, INK = "#8172B3", "#8C8C8C", "#2A2F3A"
PANEL, PEDGE = "#F7F8FA", "#D8DCE4"
LBLUE, LORANGE, LYELLOW = "#DCE6F4", "#FBE3D4", "#FFF3CD"
DGREEN, DRED = "#2F7D3B", "#9C2323"

DPI = 300
FW, FH = 12.4, 5.5   # PPT 内容区比例


def rbox(ax, x, y, w, h, text, fc="white", ec=GREY, fs=11, tc=None, bold=False,
         lw=1.2, ls="-", rs=1.4):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.4,rounding_size={rs}",
                       fc=fc, ec=ec, lw=lw, ls=ls, mutation_aspect=0.44)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or INK, fontweight="bold" if bold else "normal", linespacing=1.45)


def arr(ax, x1, y1, x2, y2, color=GREY, lw=1.6, rad=0.0, ms=15, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


def new_ax(w=FW, h=FH):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, h / w * 100)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


# ── fig1: 方法全流程 ────────────────────────────────────────────────────
def fig1():
    fig, ax = new_ax()
    H = FH / FW * 100  # 44.35

    # 三个阶段面板
    for x0, w, title in [(0.5, 35, "①  配对扰动采样"),
                         (37.5, 28.5, "②  特征提取（两模型并行）"),
                         (67.5, 32, "③  对齐评分（PCA + 逐层 ALS）")]:
        ax.add_patch(FancyBboxPatch((x0, 4.8), w, H - 5.6,
                                    boxstyle="round,pad=0.3,rounding_size=1.8",
                                    fc=PANEL, ec=PEDGE, lw=1.3, mutation_aspect=0.44))
        ax.text(x0 + w / 2, H - 2.6, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color=INK)

    # P1 竖直流程
    rbox(ax, 11, 35.5, 16, 5.2, "探针 prompt × 32\n英 / 中 / 代码 / 数学 / 问答", LBLUE, BLUE, 10.5)
    rbox(ax, 11, 26.2, 16, 5.2, "输入 embedding\n[1, T, D]", "white", GREY, 10.5)
    rbox(ax, 11, 16.9, 16, 5.2, "加高斯噪声\n$\\epsilon\\sim\\mathcal{N}(0,\\sigma^2)$,  $\\sigma=2{\\times}10^{-3}$", BLUE, BLUE, 10.5, "white", True)
    rbox(ax, 7.5, 7.2, 23, 5.6, "同一扰动复制两份（同 seed $s_k$）\nseed sets = {123, 1123, 2123}", LYELLOW, "#B8960C", 10.5, "#6B5500", True)
    arr(ax, 19, 35.5, 19, 31.6); arr(ax, 19, 26.2, 19, 22.3); arr(ax, 19, 16.9, 19, 13.0)

    # P1 → P2 两条配对弧线
    arr(ax, 30.5, 12.6, 39.2, 30.0, BLUE, 1.8, rad=0.22)
    arr(ax, 30.5, 8.0, 39.2, 9.6, ORANGE, 1.8, rad=-0.12)

    # P2 victim / suspect 两行
    rbox(ax, 39.5, 27.2, 12.8, 5.6, "victim\n7B 前向 (bf16)", LBLUE, BLUE, 10.5, BLUE, True)
    rbox(ax, 54.8, 27.2, 10.4, 5.6, "取 4 均匀层\nmask 均值池化", "white", GREY, 10)
    rbox(ax, 39.5, 6.9, 12.8, 5.6, "suspect\n7B 前向 (bf16)", LORANGE, ORANGE, 10.5, "#A34E1E", True)
    rbox(ax, 54.8, 6.9, 10.4, 5.6, "取 4 均匀层\nmask 均值池化", "white", GREY, 10)
    arr(ax, 52.3, 30.0, 54.6, 30.0); arr(ax, 52.3, 9.7, 54.6, 9.7)
    # 配对性标注（虚线椭圆）
    ax.add_patch(Ellipse((52, 20.3), 25, 8.6, fc="none", ec="#B8960C", lw=1.3, ls=(0, (5, 3))))
    ax.text(52, 20.3, "同 seed ⇒ 输入逐位相同\n特征差异 100% 归因于参数", ha="center", va="center",
            fontsize=10.5, color="#6B5500", fontweight="bold", linespacing=1.5)
    # P2 → P3
    arr(ax, 65.2, 30.0, 69.4, 35.6, BLUE, 1.6, rad=0.12)
    arr(ax, 65.2, 9.7, 69.4, 33.6, ORANGE, 1.6, rad=0.18)

    # P3 评分链
    rbox(ax, 70.5, 33.2, 26, 5.0, "PCA 降维  $d\\in\\{8, 20\\}$  (满足 $d\\leq 4N{-}1$)", "white", GREY, 10.5)
    rbox(ax, 70.5, 22.6, 26, 6.6, "逐层 ALS 线性对齐\n$W^*_l=\\mathrm{argmin}_W\\,\\|F_{\\mathrm{s}}-W F_{\\mathrm{v}}\\|^2$", "#1F3864", "#1F3864", 11, "white", True)
    rbox(ax, 70.5, 14.4, 26, 5.0, "逐层残差 $\\to$  $\\bar d = \\frac{1}{L}\\sum_l d_l$", "white", GREY, 11)
    rbox(ax, 70.5, 6.2, 26, 5.6, "similarity $= e^{-\\bar d}$   （逐 prompt 平均）", "#EAF1EA", DGREEN, 11.5, DGREEN, True)
    arr(ax, 83.5, 33.2, 83.5, 29.6); arr(ax, 83.5, 22.6, 83.5, 19.8); arr(ax, 83.5, 14.4, 83.5, 12.2)

    # 底部判定条
    rbox(ax, 0.5, 0.4, 49, 3.6, "sim $\\to$ 1：同源 —— 层表示云只差一个线性映射，ALS 可对齐",
         "#E2EFDA", DGREEN, 11, DGREEN, True)
    rbox(ax, 50.5, 0.4, 49, 3.6, "sim $\\to$ 0.25：异源地板 —— 云形状本质不同，线性对齐失效（跨家族实测）",
         "#F6D5D0", DRED, 11, DRED, True)

    out = os.path.join(FIGDIR, "fig1_method.png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out


# ── fig2: ALS 对齐原理（同源 vs 异源特征云） ───────────────────────────
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(FW, FH))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.855, bottom=0.145, wspace=0.09)
    rng = np.random.default_rng(7)
    mu_v = np.array([-2.3, 0.0])
    Sv = np.array([[1.0, 0.30], [0.30, 0.62]])
    Pv = rng.multivariate_normal(mu_v, Sv, 120)

    def ell(ax, mu, S, color, lw=1.5, ls="-", n=1.6):
        vals, vecs = np.linalg.eigh(S)
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(mu, 2 * n * np.sqrt(vals[1]), 2 * n * np.sqrt(vals[0]),
                             angle=ang, fc="none", ec=color, lw=lw, ls=ls, zorder=4))

    titles = ["(a) 同源：只差一个线性映射 —— ALS 可对齐",
              "(b) 异源：云形状本质不同 —— 线性对齐失效"]
    for k, ax in enumerate(axes):
        ax.set_xlim(-6.6, 6.6); ax.set_ylim(-4.4, 4.4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#C4CAD4"); sp.set_linewidth(1.2)
        ax.scatter(Pv[:, 0], Pv[:, 1], s=21, c=BLUE, alpha=0.78,
                   edgecolors="white", lw=0.35, zorder=3)
        ell(ax, mu_v, Sv, BLUE)
        ax.text(mu_v[0], mu_v[1] - 2.6, "victim 特征云", ha="center", fontsize=11, color=BLUE)
        if k == 0:   # (a) suspect = 线性变换后的 victim 云（fine-tune 漂移）
            A = np.array([[1.28, 0.36], [-0.20, 0.97]])
            Ps = (Pv - mu_v) @ A.T + np.array([2.6, 0.3]) + rng.normal(0, 0.04, Pv.shape)
            ax.scatter(Ps[:, 0], Ps[:, 1], s=21, c=ORANGE, alpha=0.8,
                       edgecolors="white", lw=0.35, zorder=3)
            ell(ax, Ps.mean(0), A @ Sv @ A.T, ORANGE)
            arr(ax, -1.35, 1.15, 1.35, 1.45, GREY, 2.0, rad=0.30, ms=17)
            ax.text(0.0, 2.75, "$W^{*}$", fontsize=15, color=INK, fontweight="bold")
            ell(ax, mu_v + np.array([0.12, -0.10]), Sv, "#9AA2B2", lw=1.3, ls=(0, (4, 3)))
            ax.text(-2.3, -3.6, "对齐后残差（小）", fontsize=10.5, color="#5A6272", ha="center")
            ax.text(3.4, 3.35, "same origin（fine_tune）\n$\\bar{d}$ = 1.36 → 高相似",
                    ha="center", va="center", fontsize=11.5, color=DGREEN, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.45", fc="#EAF3EA", ec=DGREEN, lw=1.1))
        else:        # (b) suspect 形状本质不同（cross-family）
            th = np.radians(66)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            Sb = R @ np.diag([0.30, 3.6]) @ R.T
            Ps = rng.multivariate_normal(np.array([2.5, -0.35]), Sb, 120)
            ax.scatter(Ps[:, 0], Ps[:, 1], s=21, c=ORANGE, alpha=0.8,
                       edgecolors="white", lw=0.35, zorder=3)
            ell(ax, np.array([2.5, -0.35]), Sb, ORANGE)
            arr(ax, -1.35, 1.15, 1.45, 0.6, GREY, 2.0, rad=0.22, ms=17)
            cx, cy = 2.5, -0.35
            for dx in (-1, 1):
                ax.plot([cx - 0.42 * dx, cx + 0.42 * dx], [cy - 0.42, cy + 0.42],
                        color=RED, lw=2.6, zorder=6, solid_capstyle="round")
            ax.text(3.4, 3.35, "cross-family（Qwen2.5-7B）\n$\\bar{d}$ = 5.58 → 地板 sim≈0.25",
                    ha="center", va="center", fontsize=11.5, color=DRED, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.45", fc="#F9E4E2", ec=DRED, lw=1.1))
        ax.set_title(titles[k], fontsize=13.5, fontweight="bold", color=INK, pad=10)
    fig.text(0.5, 0.032,
             "核心假设：同源模型的层表示云只差一个线性映射 —— 拟合 $W$ 后残差即距离 $\\bar{d}$；"
             "置换 / 各向异性缩放属于线性映射 ⇒ 白盒伪装必然隐形。（$\\bar{d}$ 取 N=128, d=8 实测均值）",
             ha="center", fontsize=11.5, color=INK)
    out = os.path.join(FIGDIR, "fig2_als.png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out


# ── fig3: 两线谱系面板 ──────────────────────────────────────────────────
def fig3():
    fig, ax = new_ax()
    # 主线
    ax.text(1.5, 41.8, "主线 — 谱系判别", fontsize=13, fontweight="bold", color=BLUE, ha="left")
    rbox(ax, 36, 34.4, 28, 5.4, "victim — Llama-2-7b-hf（32 层 / D=4096）", "#1F3864", "#1F3864", 11, "white", True)
    rbox(ax, 1.5, 26.2, 26, 5.4, "fine_tune — Llama-2-7b-chat\n（官方 SFT 正例）", LBLUE, BLUE, 10.5, BLUE, True)
    rbox(ax, 37, 26.2, 26, 5.4, "derivative_cn — Chinese-llama-2-7b\n（独立衍生正例）", "#E8E2F2", PURPLE, 10.5, "#5D4B8B")
    rbox(ax, 72.5, 26.2, 26, 5.4, "同底座剪枝难负例 ×3\nsparsegpt / wanda / gblm", LORANGE, ORANGE, 10.5, "#A34E1E", True)
    arr(ax, 46, 34.4, 14.5, 31.8, GREY, 1.6, rad=0.22)
    arr(ax, 50, 34.4, 50, 31.8, GREY, 1.6)
    arr(ax, 54, 34.4, 85.5, 31.8, GREY, 1.6, rad=-0.22)
    rbox(ax, 1.5, 17.4, 26, 5.2, "白盒伪装 ×3（免采集）\nweight_perm / rep_perm / scale ＋ prune_dpoff", LYELLOW, "#B8960C", 10, "#6B5500", ls=(0, (4, 2.5)))
    rbox(ax, 72.5, 17.4, 26, 5.2, "第三方真实剪枝：保留同底座几何\n→ 预期更靠近 victim（关键新维度）", "#FDF0E6", ORANGE, 10, "#A34E1E", ls=(0, (4, 2.5)))
    arr(ax, 14.5, 26.2, 14.5, 22.8, "#B8960C", 1.4, ls=(0, (4, 2.5)))
    arr(ax, 85.5, 26.2, 85.5, 22.8, ORANGE, 1.4, ls=(0, (4, 2.5)))
    rbox(ax, 1.5, 10.4, 97, 4.6, "易负例（跨家族）：Qwen2.5-7B ｜ Meta-Llama-3-8B ｜ internlm2-7b（数值发散 → 已隔离）", "#E2EFDA", GREEN, 10.5, DGREEN)
    # 蒸馏线（全直线布局，避免连线穿插）
    ax.text(1.5, 9.7, "蒸馏线 — 蒸馏指纹检测（双臂对照）", fontsize=13, fontweight="bold",
            color="#B05A1E", ha="left")
    rbox(ax, 1.5, 2.8, 17, 5.4, "victim — Llama-3.2-1B\n（16 层 / D=2048）", "#1F3864", "#1F3864", 10, "white", True)
    rbox(ax, 24.5, 4.5, 25, 4.8, "student_distill\n3B 教师回答 · 序列级 CE · 1600×3ep", LORANGE, ORANGE, 10, "#A34E1E", True)
    rbox(ax, 54, 1.1, 25, 4.8, "student_control\nAlpaca 原始回答 · 同预算（关键对照）", LYELLOW, "#B8960C", 10, "#6B5500", True)
    rbox(ax, 84, 4.5, 14.5, 4.8, "Llama-3.2-3B\n教师（base）\n生成 1600 条", "#E3E9F2", BLUE, 9.5, BLUE)
    arr(ax, 18.5, 7.0, 24.2, 7.0, GREY, 1.6)
    arr(ax, 18.5, 3.6, 53.7, 3.5, GREY, 1.6, rad=-0.06)
    arr(ax, 84, 7.0, 49.6, 7.0, BLUE, 1.7)
    ax.text(66.9, 7.55, "教师回答", fontsize=10, color=BLUE, ha="center")
    ax.text(13, 1.15, "双臂唯一差异 = 训练文本 ⇒ 蒸馏净效应 = distill − control",
            fontsize=10.5, color="#5A6272", ha="left", va="center")

    out = os.path.join(FIGDIR, "fig3_panel.png")
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out


def make_all():
    return [fig1(), fig2(), fig3()]


if __name__ == "__main__":
    for p in make_all():
        print("saved", p)
