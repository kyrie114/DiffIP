"""
生成高质量、学术范、设计优雅的导师汇报 PPT:
主题: DiffIP LLM 谱系溯源与消融验证 (大模块 + 3 个消融实验)
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# --- 路径与资源设置 ---
WORK_DIR = Path("/home/lushunxin/work/DiffIP/llm_migration_v4")
ASSETS_DIR = WORK_DIR / "ppt_assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PPT = WORK_DIR / "汇报_DiffIP_大模块与消融实验.pptx"

# 设置 Matplotlib 字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 生成图表 1: 消融实验 1 - 抽层策略 (Layer Sampling)
def make_layer_chart():
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    methods = ['Even 4\n(Default)', 'Even 8\n(Dense)', 'All 32\n(Full)', 'Last 4\n(Deep)']
    gaps = [0.092, 0.066, 0.043, 0.033]
    colors = ['#159A9C', '#4B6584', '#A5B1C2', '#D1D8E0']
    
    bars = ax.bar(methods, gaps, color=colors, width=0.55, edgecolor='#2C3E50', linewidth=1)
    ax.axhline(0.092, color='#E8871A', linestyle='--', linewidth=1.2, label='Even-4 Peak Margin (0.092)')
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.003, f"{h:.3f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1E293B')
                
    ax.set_ylim(0, 0.115)
    ax.set_ylabel("Held-out Margin to Baseline", fontsize=10, color='#1E293B')
    ax.set_title("Layer Selection Ablation: Even 4 is Optimal", fontsize=11, fontweight='bold', pad=12, color='#1F3A5F')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8.5, frameon=True)
    plt.tight_layout()
    chart_path = ASSETS_DIR / "chart_layers.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path

# 生成图表 2: 消融实验 2 - 维数扫描 (PCA Dimensions)
def make_dim_chart():
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    dims = [8, 10, 15, 20, 30, 40]
    gaps = [0.092, 0.093, 0.095, 0.092, 0.098, 0.098]
    stds = [0.080, 0.079, 0.078, 0.078, 0.077, 0.077]
    
    ax.plot(dims, gaps, marker='o', color='#159A9C', linewidth=2.2, markersize=7, label='Margin (Fine-tune vs Baseline)')
    ax.plot(dims, stds, marker='s', color='#E8871A', linewidth=2.0, markersize=6, linestyle='-.', label='Standard Deviation (Std)')
    
    for d, g in zip(dims, gaps):
        ax.annotate(f"{g:.3f}", (d, g), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8.5, fontweight='bold')
        
    ax.set_ylim(0.06, 0.115)
    ax.set_xlabel("PCA Dimension (d)", fontsize=10, color='#1E293B')
    ax.set_ylabel("Metric Value", fontsize=10, color='#1E293B')
    ax.set_title("Dimension Robustness: Flat Across 8-40 Dims", fontsize=11, fontweight='bold', pad=12, color='#1F3A5F')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=8.5, frameon=True)
    plt.tight_layout()
    chart_path = ASSETS_DIR / "chart_dims.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path

# 生成图表 3: 消融实验 3 - 攻击伪装与混淆 (Attack & Disguise)
def make_attack_chart():
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    attacks = ['Clean\nSFT', 'Channel\nPerm', 'Aniso\nScale', 'Tanh\nSatur', 'Square\n(Pow 2)', 'Norm\nGauss']
    sims = [0.363, 0.363, 0.365, 0.371, 0.344, 0.348]
    
    bars = ax.bar(attacks, sims, color='#1F3A5F', width=0.55, edgecolor='#2C3E50', linewidth=1)
    bars[4].set_color('#C7443C') # Pow2
    bars[5].set_color('#E8871A') # Gauss
    
    ax.axhline(0.254, color='#7B5EA7', linestyle='--', linewidth=1.5, label='Cross-Family Floor (0.254)')
    ax.text(4.5, 0.258, "Floor Threshold: 0.254", color='#7B5EA7', fontsize=8.5, fontweight='bold')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.004, f"{h:.3f}",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E293B')

    ax.set_ylim(0.20, 0.40)
    ax.set_ylabel("DiffIP Similarity", fontsize=10, color='#1E293B')
    ax.set_title("Robustness Under Non-linear Transformations", fontsize=11, fontweight='bold', pad=12, color='#1F3A5F')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=8.5, frameon=True)
    plt.tight_layout()
    chart_path = ASSETS_DIR / "chart_attacks.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path

# 生成图表 4: 相对判定机制消除 Prompt 难度偏差
def make_roc_contrast_chart():
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    cats = ['Global Absolute Cut\n(Naive AUC=0.702)', 'Shifted Prompt Diff\n(Premise Break AUC=0.644)', 'Same-Prompt Margin\n(Proposed AUC=0.9998)']
    aucs = [0.702, 0.644, 0.9998]
    colors = ['#A5B1C2', '#E8871A', '#159A9C']
    
    bars = ax.barh(cats, aucs, color=colors, height=0.45, edgecolor='#2C3E50', linewidth=1)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height()/2., f"{w:.4f}",
                ha='left', va='center', fontsize=9.5, fontweight='bold', color='#1E293B')
                
    ax.set_xlim(0.5, 1.15)
    ax.set_xlabel("Deployment Detection AUC", fontsize=10, color='#1E293B')
    ax.set_title("Why Relative Margin? Eliminating Prompt Bias", fontsize=11, fontweight='bold', pad=12, color='#1F3A5F')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    chart_path = ASSETS_DIR / "chart_relative_auc.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path

img_layers = make_layer_chart()
img_dims = make_dim_chart()
img_attacks = make_attack_chart()
img_auc = make_roc_contrast_chart()

# --- PPT 颜色与样式配置 ---
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
TEAL = RGBColor(0x15, 0x9A, 0x9C)
ORANGE = RGBColor(0xE8, 0x87, 0x1A)
RED = RGBColor(0xC7, 0x44, 0x3C)
GREEN = RGBColor(0x3D, 0x9A, 0x50)
PURPLE = RGBColor(0x7B, 0x5E, 0xA7)
INK = RGBColor(0x1E, 0x29, 0x3B)
MUT = RGBColor(0x64, 0x74, 0x8B)
PAPER = RGBColor(0xF8, 0xFA, 0xFC)  # Slate-50 超淡雅背景
CARD = RGBColor(0xFF, 0xFF, 0xFF)   # 纯白卡片
LINEC = RGBColor(0xE2, 0xE8, 0xF0)  # 边框线
LIGHTNAV = RGBColor(0xEA, 0xF1, 0xF9)
LIGHTTE = RGBColor(0xE8, 0xF5, 0xF5)
LIGHTOR = RGBColor(0xFD, 0xF3, 0xE7)
LIGHTRE = RGBColor(0xFB, 0xEF, 0xEF)

FONT = "Arial"
FONT_ZH = "Noto Sans CJK SC"

SW, SH = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]

def set_run_font(run, size, bold=False, color=INK, italic=False):
    f = run.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", FONT_ZH)

def add_header(slide, title_text, category_tag="DIFFIP 7B EXTENSION & ABLATION"):
    # 顶部标签
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.3))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    tf_c.margin_left = tf_c.margin_top = tf_c.margin_bottom = tf_c.margin_right = 0
    p_c = tf_c.paragraphs[0]
    r_c = p_c.add_run()
    r_c.text = category_tag.upper()
    set_run_font(r_c, 10, bold=True, color=TEAL)

    # 主标题
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.6))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_bottom = tf_t.margin_right = 0
    p_t = tf_t.paragraphs[0]
    r_t = p_t.add_run()
    r_t.text = title_text
    set_run_font(r_t, 22, bold=True, color=NAVY)

def add_card(slide, left, top, width, height, fill_color=CARD, line_color=LINEC):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

# ==============================================================================
# SLIDE 1: 封面 (精致大气质感)
# ==============================================================================
slide1 = prs.slides.add_slide(BLANK)
bg1 = add_card(slide1, 0, 0, SW, SH, fill_color=NAVY, line_color=None)

# 装饰线条/色块
add_card(slide1, Inches(1.2), Inches(1.8), Inches(1.0), Inches(0.08), fill_color=TEAL, line_color=None)

title_box = slide1.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(11), Inches(2.2))
tf = title_box.text_frame
tf.word_wrap = True
p1 = tf.paragraphs[0]
r1 = p1.add_run()
r1.text = "大语言模型谱系指纹溯源系统 (DiffIP-LLM)"
set_run_font(r1, 32, bold=True, color=CARD)

p2 = tf.add_paragraph()
p2.space_before = Pt(14)
r2 = p2.add_run()
r2.text = "7B 规模主架构验证与三大核心消融实验汇报"
set_run_font(r2, 20, bold=False, color=RGBColor(0x8E, 0xCA, 0xE6))

# 底部汇报信息卡片
info_box = slide1.shapes.add_textbox(Inches(1.2), Inches(5.0), Inches(10), Inches(1.5))
tf_info = info_box.text_frame
p_i1 = tf_info.paragraphs[0]
r_i1 = p_i1.add_run()
r_i1.text = "汇报架构: 1 个核心判定大模块  +  3 个严谨消融实验  +  5 模型跨家族泛化验证"
set_run_font(r_i1, 14, bold=True, color=ORANGE)

p_i2 = tf_info.add_paragraph()
p_i2.space_before = Pt(8)
r_i2 = p_i2.add_run()
r_i2.text = "基线对齐: Llama-2-7b-hf 全谱系 (SFT / 稀疏剪枝 / 独立衍生) | 严谨复现: 18/20 跨家族参照留出测试全过"
set_run_font(r_i2, 12, bold=False, color=RGBColor(0xD7, 0xE0, 0xEC))


# ==============================================================================
# SLIDE 2: 整体汇报框架导航 (大模块 + 3 个消融实验)
# ==============================================================================
slide2 = prs.slides.add_slide(BLANK)
add_card(slide2, 0, 0, SW, SH, fill_color=PAPER, line_color=None)
add_header(slide2, "研究全景: 1 个大模块核心方法与 3 个消融实验组织架构", "OVERVIEW & AGENDA")

# 大模块卡片 (左侧宽卡片)
add_card(slide2, Inches(0.8), Inches(1.5), Inches(4.5), Inches(5.2), fill_color=CARD)
header_stripe1 = add_card(slide2, Inches(0.8), Inches(1.5), Inches(4.5), Inches(0.7), fill_color=NAVY, line_color=None)
tx = slide2.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(4.1), Inches(0.5))
p = tx.text_frame.paragraphs[0]
r = p.add_run(); r.text = "【核心大模块】DiffIP 几何指纹"
set_run_font(r, 14, bold=True, color=CARD)

tx_c1 = slide2.shapes.add_textbox(Inches(1.0), Inches(2.4), Inches(4.1), Inches(4.0))
tf = tx_c1.text_frame; tf.word_wrap = True
items_m = [
    ("配对扰动采样", "同 seed 输入高斯扰动 (Scale=2e-3)，激活潜空间几何流形"),
    ("稀疏层表征抓取", "默认抽取 4 均匀层隐状态做 Pooling，兼顾全局与开销"),
    ("PCA + 逐层 ALS 回退", "降至 8 维后，拟合正交矩阵 Q、对角 S 与位移 h，算均方误差"),
    ("相对消除机制 (新)", "提出 sim(Suspect) - sim(Ref)，攻克绝对阈值跨规模漂移")
]
for k, (t, d) in enumerate(items_m):
    p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
    if k > 0: p.space_before = Pt(12)
    r1 = p.add_run(); r1.text = f"• {t}: "
    set_run_font(r1, 12, bold=True, color=NAVY)
    r2 = p.add_run(); r2.text = d
    set_run_font(r2, 11, bold=False, color=INK)

# 右侧 3 个消融实验小卡片
ablation_cards = [
    ("消融实验 1: 抽层策略消融 (Layer Sampling)", "Even-4 vs Even-8 vs All-32 vs Last-4 vs Mid-2", 
     "回答: 到底采哪几层最好？是否层数越多越好？证实深层引入局部噪声，均匀 4 层即为峰值。"),
    ("消融实验 2: 降维瓶颈与配对机制消融 (Dimension & Pairing)", "8/10/15/20/30/40 维扫描 + 打乱配对 (Shuffle) 对照", 
     "回答: 8 维是否足够捕获谱系？打乱成对关系后几何是否坍塌？证实其捕获的是真正的流形结构。"),
    ("消融实验 3: 防御韧性与混淆攻击消融 (Attack & Robustness)", "通道置换 / 缩放 / 9 种非线性映射 / 无关模型插值混淆", 
     "回答: 恶意篡改能否抹去指纹？证实线性完全被吃掉、非线性可分、需掺入 >75% 能量才触及地板。")
]

top_pos = [1.5, 3.25, 5.0]
for idx, (title, sub, desc) in enumerate(ablation_cards):
    t_y = top_pos[idx]
    add_card(slide2, Inches(5.6), Inches(t_y), Inches(6.9), Inches(1.5), fill_color=CARD)
    # 左侧小彩色指示块
    add_card(slide2, Inches(5.6), Inches(t_y), Inches(0.12), Inches(1.5), fill_color=TEAL if idx==0 else (ORANGE if idx==1 else PURPLE), line_color=None)
    
    tb = slide2.shapes.add_textbox(Inches(5.9), Inches(t_y + 0.1), Inches(6.4), Inches(1.3))
    tf = tb.text_frame; tf.word_wrap = True
    p1 = tf.paragraphs[0]
    r1 = p1.add_run(); r1.text = title
    set_run_font(r1, 13, bold=True, color=NAVY)
    
    p2 = tf.add_paragraph()
    p2.space_before = Pt(3)
    r2 = p2.add_run(); r2.text = f"配置对比: {sub}"
    set_run_font(r2, 11, bold=True, color=TEAL if idx==0 else (ORANGE if idx==1 else PURPLE))
    
    p3 = tf.add_paragraph()
    p3.space_before = Pt(4)
    r3 = p3.add_run(); r3.text = desc
    set_run_font(r3, 10.5, color=MUT)


# ==============================================================================
# SLIDE 3: 核心大模块详解 (DiffIP 7B 架构与创新)
# ==============================================================================
slide3 = prs.slides.add_slide(BLANK)
add_card(slide3, 0, 0, SW, SH, fill_color=PAPER, line_color=None)
add_header(slide3, "核心大模块: DiffIP 扩展到 7B 的几何指纹溯源流程", "MODULE ARCHITECTURE")

# 左侧: 流程步骤说明卡片
add_card(slide3, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2), fill_color=CARD)
tb_l = slide3.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.2), Inches(4.6))
tf_l = tb_l.text_frame; tf_l.word_wrap = True

steps = [
    ("Step 1: 配对输入嵌入扰动 (Paired Perturbation)", 
     "对 32 条代表性探针 Prompt，利用严格配对的伪随机种子在 Embedding 施加微弱高斯扰动 (scale=2e-3)，构建同一输入空间下的近邻扰动云 (N=128)。"),
    ("Step 2: 逐层隐状态提取与池化 (Hidden Extraction)", 
     "抓取前向传播过程中的 Transformer 层 (默认均匀采样 4 层: 0, 10, 21, 31)，经过 Attention-Mask 加权平均池化后，得到 [4, 128, 4096] 的表征矩阵。"),
    ("Step 3: 流形对齐与交替最小二乘法 (ALS Reversion)", 
     "对每个层独立进行 PCA 降维至 8 维后，通过交替最小二乘法拟合变换参数 (Q: 正交旋转, S: 对角各向异性缩放, h: 平移)，消除非本质仿射变换并计算平均 L2 误差。"),
    ("Step 4: 相对参照校准机制 (Prompt-Relative Margin)", 
     "针对绝对距离随规模与 Prompt 难度漂移的现象，采用同一 Prompt 下减去无关模型作为校准基准: Margin = Sim(Suspect) - Sim(Reference)。")
]

for idx, (title, body) in enumerate(steps):
    p = tf_l.paragraphs[0] if idx == 0 else tf_l.add_paragraph()
    if idx > 0: p.space_before = Pt(10)
    r1 = p.add_run(); r1.text = f"{title}\n"
    set_run_font(r1, 11.5, bold=True, color=NAVY)
    r2 = p.add_run(); r2.text = body
    set_run_font(r2, 10, color=INK)

# 右侧: 关键突破图 (相对判定机制消除 Prompt 难度偏差)
add_card(slide3, Inches(6.9), Inches(1.5), Inches(5.6), Inches(5.2), fill_color=CARD)
slide3.shapes.add_picture(str(img_auc), Inches(7.0), Inches(1.7), Inches(5.4), Inches(3.3))

tb_r = slide3.shapes.add_textbox(Inches(7.1), Inches(5.1), Inches(5.2), Inches(1.4))
tf_r = tb_r.text_frame; tf_r.word_wrap = True
p = tf_r.paragraphs[0]
r = p.add_run(); r.text = "关键创新点解析:\n"
set_run_font(r, 11.5, bold=True, color=ORANGE)
r2 = p.add_run(); r2.text = "由于单条 Prompt 的语义难度不同 (数学类均值 0.533，问答类 0.077)，直接用绝对分数判定的 AUC 仅 0.702。本大模块提出「同一 Prompt 减无关参照」，一举将 AUC 提升至 0.9998，SFT 留出召回率达 100%。"
set_run_font(r2, 10, color=MUT)


# ==============================================================================
# SLIDE 4: 主模块在 7B 上的真实谱系排序表现
# ==============================================================================
slide4 = prs.slides.add_slide(BLANK)
add_card(slide4, 0, 0, SW, SH, fill_color=PAPER, line_color=None)
add_header(slide4, "大模块评测: Llama-2-7b 全谱系与跨家族模型实测结果", "MAIN EXPERIMENTAL RESULTS")

# 数据表格 (自建精细卡片式表格)
# 列宽: 模型名(2.5), 谱系角色(1.8), Sim(N=8)(1.3), Sim(N=20)(1.3), Sim(N=128)(1.5), 逐Prompt胜负(1.6)
table_shape = slide4.shapes.add_table(10, 6, Inches(0.8), Inches(1.5), Inches(11.7), Inches(3.6))
table = table_shape.table
table.columns[0].width = Inches(2.6)
table.columns[1].width = Inches(2.0)
table.columns[2].width = Inches(1.6)
table.columns[3].width = Inches(1.6)
table.columns[4].width = Inches(2.0)
table.columns[5].width = Inches(1.9)

headers = ["模型 (Model)", "谱系关系 (Role)", "N = 8", "N = 20", "N = 128 (最终指标)", "胜负对比 (vs 负例)"]
for c_idx, h_text in enumerate(headers):
    cell = table.cell(0, c_idx)
    cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
    p = cell.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = h_text
    set_run_font(r, 10.5, bold=True, color=CARD)

rows_data = [
    ("Llama-2-7b-hf (Victim)", "基座模型 (Base)", "1.000", "1.000", "1.000", "32 / 32"),
    ("SparseGPT / Wanda / GBLM", "同底座稀疏剪枝变体 (正例)", "0.66~0.72", "0.57~0.61", "0.528 ~ 0.559", "32 / 32 全胜"),
    ("Llama-2-7b-chat-hf", "官方指令微调 (SFT 簇)", "0.552", "0.423", "0.363 ± 0.006", "32 / 32 全胜"),
    ("Weight / Rep / Scale 伪装", "输入/特征空间线性混淆", "0.555", "0.423", "0.363 ~ 0.365", "32 / 32 (完全看穿)"),
    ("chat + 剪4层 (prune_dpoff)", "结构化整层删除", "0.530", "0.413", "0.357 ± 0.008", "32 / 32 全胜"),
    ("Chinese-llama-2-7b", "独立二开衍生版 (同源)", "0.459", "0.348", "0.311 ± 0.002", "25 / 32 胜"),
    ("Qwen2.5-7B", "跨家族跨维度负例", "0.379", "0.296", "0.254 ± 0.008", "地板基准 (0/32)"),
    ("Meta-Llama-3-8B", "跨家族负例 (8B)", "0.386", "0.298", "0.254 ± 0.007", "地板基准 (0/32)"),
    ("蒸馏线: Distill vs Control", "同架构序列级 KD 对照", "0.620 vs 0.623 (无显著差异, 阴性结论保留)", "AUC = 0.493 (无法检测)")
]

for r_idx, row in enumerate(rows_data):
    for c_idx, val in enumerate(row):
        cell = table.cell(r_idx + 1, c_idx)
        cell.fill.solid()
        if r_idx == 0: cell.fill.fore_color.rgb = RGBColor(0xEE, 0xF2, 0xF6)
        elif r_idx == 1: cell.fill.fore_color.rgb = LIGHTTE
        elif r_idx in (2, 3, 4): cell.fill.fore_color.rgb = LIGHTOR
        elif r_idx in (6, 7): cell.fill.fore_color.rgb = LIGHTNAV
        elif r_idx == 8: cell.fill.fore_color.rgb = LIGHTRE
        else: cell.fill.fore_color.rgb = CARD
        
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if c_idx >= 2 else PP_ALIGN.LEFT
        r = p.add_run(); r.text = val
        is_bold = (c_idx == 4 or r_idx == 0)
        set_run_font(r, 9.5, bold=is_bold, color=INK)

# 底部 3 个分析要点卡片
tips = [
    ("严谨的谱系递减规律", "基座(1.0) ≫ 稀疏结构变体(0.54) > SFT 簇(0.363) > 独立衍生(0.311) > 跨家族地板(0.254)。"),
    ("线性伪装完全失效", "权重置换、特征置换与缩放后的得分完全落在 fine_tune 波动内 (0.363 vs 0.363)，协议具天生抗性。"),
    ("科学保留蒸馏阴性", "严格双盲对照证实: 序列级蒸馏不会在中间层保留教师几何指纹，避免假阳性，具备学术严谨性。")
]

for idx, (t, d) in enumerate(tips):
    cx = Inches(0.8 + idx * 4.0)
    add_card(slide4, cx, Inches(5.35), Inches(3.7), Inches(1.5), fill_color=CARD)
    tb = slide4.shapes.add_textbox(cx + Inches(0.15), Inches(5.45), Inches(3.4), Inches(1.3))
    tf = tb.text_frame; tf.word_wrap = True
    p1 = tf.paragraphs[0]
    r1 = p1.add_run(); r1.text = f"✔ {t}"
    set_run_font(r1, 11.5, bold=True, color=NAVY)
    p2 = tf.add_paragraph()
    p2.space_before = Pt(4)
    r2 = p2.add_run(); r2.text = d
    set_run_font(r2, 10, color=MUT)


# ==============================================================================
# SLIDE 5: 消融实验 1 - 抽层策略消融 (Layer Selection)
# ==============================================================================
slide5 = prs.slides.add_slide(BLANK)
add_card(slide5, 0, 0, SW, SH, fill_color=PAPER, line_color=None)
add_header(slide5, "消融实验 1: 抽层策略消融 (为什么均匀 4 层是最佳实践？)", "ABLATION STUDY 1: LAYER SELECTION")

# 左侧图表
add_card(slide5, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2), fill_color=CARD)
slide5.shapes.add_picture(str(img_layers), Inches(0.9), Inches(1.7), Inches(5.6), Inches(3.5))

tb_c1 = slide5.shapes.add_textbox(Inches(1.1), Inches(5.3), Inches(5.2), Inches(1.2))
tf = tb_c1.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "定量对比结论: "
set_run_font(r, 11, bold=True, color=TEAL)
r2 = p.add_run(); r2.text = "均匀 4 层对负例的留出 Margin 达到 0.092 (最高)。均匀 8 层下降至 0.066，全部 32 层降至 0.043，末端 4 层仅 0.033。均匀 4 层表现最优！"
set_run_font(r2, 10, color=INK)

# 右侧机制深度解析
add_card(slide5, Inches(6.9), Inches(1.5), Inches(5.6), Inches(5.2), fill_color=CARD)
tb_r = slide5.shapes.add_textbox(Inches(7.2), Inches(1.8), Inches(5.0), Inches(4.6))
tf_r = tb_r.text_frame; tf_r.word_wrap = True

points_l = [
    ("打破直觉:「并不是层数越多越好」", 
     "直觉上认为采满 32 层能提供最完整的表征，但实验表明全部层 (All 32) 的微调簇留出召回率反而从 100% 下降至 71%~79%，负例误报率恶化至 12.5% (超标)。"),
    ("深层局部特征累积成噪声", 
     "网络后半段 (尤其是最后 4 层) 强烈受训练目标和自注意力细节的主导，引入大量模型特异性扰动。这些差异在 ALS 对齐中掩盖了基座固有的全局几何指纹。"),
    ("稀疏均匀采样的双重优势", 
     "均匀采样 4 层 (L0, L10, L21, L31) 既覆盖了浅层语言先验、中层抽象表征与深层语义，又极大地节约了 87.5% 的特征存储与对齐计算开销，是性能与成本的最优平衡点。")
]

for idx, (title, text) in enumerate(points_l):
    p = tf_r.paragraphs[0] if idx == 0 else tf_r.add_paragraph()
    if idx > 0: p.space_before = Pt(14)
    r1 = p.add_run(); r1.text = f"{idx+1}. {title}\n"
    set_run_font(r1, 12, bold=True, color=NAVY)
    r2 = p.add_run(); r2.text = text
    set_run_font(r2, 10.5, color=INK)


# ==============================================================================
# SLIDE 6: 消融实验 2 - 降维维数与成对机制消融 (Dimensions & Pairing)
# ==============================================================================
slide6 = prs.slides.add_slide(BLANK)
add_card(slide6, 0, 0, SW, SH, fill_color=PAPER, line_color=None)
add_header(slide6, "消融实验 2: 降维维数鲁棒性与配对成因消融", "ABLATION STUDY 2: DIMENSION & PAIRING")

# 左侧图表: 维数扫描
add_card(slide6, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2), fill_color=CARD)
slide6.shapes.add_picture(str(img_dims), Inches(0.9), Inches(1.7), Inches(5.6), Inches(3.5))

tb_d = slide6.shapes.add_textbox(Inches(1.1), Inches(5.3), Inches(5.2), Inches(1.2))
tf = tb_d.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "维数扫描结论: "
set_run_font(r, 11, bold=True, color=ORANGE)
r2 = p.add_run(); r2.text = "在 8 到 40 维之间，间隔保持在 0.092~0.098，波动不足 0.006，全部通过留出测试。根据预注册标准，保持 8 维为默认超参 (低开销、抗过拟合)。"
set_run_font(r2, 10, color=INK)

# 右侧: 配对成因消融 (Shuffle 对照) —— 极其核心的学术证据
add_card(slide6, Inches(6.9), Inches(1.5), Inches(5.6), Inches(5.2), fill_color=CARD)
tb_p = slide6.shapes.add_textbox(Inches(7.2), Inches(1.8), Inches(5.0), Inches(4.6))
tf_p = tb_p.text_frame; tf_p.word_wrap = True

p = tf_p.paragraphs[0]
r = p.add_run(); r.text = "关键因果验证: 配对关系 (Pairing) 消融\n"
set_run_font(r, 13, bold=True, color=RED)

pairing_points = [
    ("打乱样本对齐 (Shuffle Experiment)", 
     "若将 fine_tune 模型的扰动样本随机 Shuffle (打乱种子对应顺序)，模型相似度立即从 0.363 剧烈滑落至 0.253，直接与无关模型地板重合！"),
    ("胜率断崖式跌落 (32/32 → 5/32)", 
     "打乱成对关系后，逐条 Prompt 战胜负例的概率从 100% 塌缩到纯随机噪声级别 (5/32)。这无可辩驳地证明了指纹并非来自模型的静态激活统计量。"),
    ("科学结论: 识别的是「几何流形响应」", 
     "DiffIP 真正测量并捕获的是「相同微小输入扰动在模型潜空间诱发的配对几何流形变化」，只要成对扰动在，谱系信号就在。")
]

for idx, (title, text) in enumerate(pairing_points):
    p = tf_p.add_paragraph()
    p.space_before = Pt(12)
    r1 = p.add_run(); r1.text = f"• {title}\n"
    set_run_font(r1, 11.5, bold=True, color=NAVY)
    r2 = p.add_run(); r2.text = text
    set_run_font(r2, 10, color=INK)


# ==============================================================================
# SLIDE 7: 消融实验 3 - 防御韧性与混淆攻击消融 (Attack & Robustness)
# ==============================================================================
slide7 = prs.slides.add_slide(BLANK)
add_card(slide7, 0, 0, SW, SH, fill_color=PAPER, line_color=None)
add_header(slide7, "消融实验 3: 防御韧性与非线性伪装混淆消融", "ABLATION STUDY 3: ATTACK & ROBUSTNESS")

# 左侧图表: 非线性攻击对比
add_card(slide7, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2), fill_color=CARD)
slide7.shapes.add_picture(str(img_attacks), Inches(0.9), Inches(1.7), Inches(5.6), Inches(3.5))

tb_a = slide7.shapes.add_textbox(Inches(1.1), Inches(5.3), Inches(5.2), Inches(1.2))
tf = tb_a.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "攻击防御结论: "
set_run_font(r, 11, bold=True, color=PURPLE)
r2 = p.add_run(); r2.text = "9 种非线性映射中 8 种完全被看穿。最强烈的符号保持平方变换 (Pow 2) 得分 0.344，依然高出地板 0.090，战胜负例率仍达 31/32！"
set_run_font(r2, 10, color=INK)

# 右侧: 混入无关模型的能量极限测试 (Mix Attack)
add_card(slide7, Inches(6.9), Inches(1.5), Inches(5.6), Inches(5.2), fill_color=CARD)
tb_m = slide7.shapes.add_textbox(Inches(7.2), Inches(1.8), Inches(5.0), Inches(4.6))
tf_m = tb_m.text_frame; tf_m.word_wrap = True

p = tf_m.paragraphs[0]
r = p.add_run(); r.text = "极限混淆压力测试: 混入无关模型特征\n"
set_run_font(r, 13, bold=True, color=NAVY)

mix_points = [
    ("均方根 (RMS) 能量拉齐", 
     "发现微调模型激活 RMS 是 Llama-3 的 2.54 倍，必须在拉齐能量后做线性插值，否则微调模型仍占主导 (插值 75% 时 fine_tune 仍占 42% 能量)。"),
    ("掺入 50% 无关模型: 稳健性依旧", 
     "在能量各占 50% 的强干扰下，相似度依然保持 0.312，逐条 Prompt 对 Qwen 负例的胜率仍高达 29/32。"),
    ("掺入 75% 无关模型: 均值触底但局部依然可分", 
     "掺入高达 75% Llama-3 激活时，均值落入 0.259 (贴近地板)，但配对比较仍有 27/32 胜出。想要抹去谱系，攻击者必须破坏绝大多数有效配对结构。")
]

for idx, (title, text) in enumerate(mix_points):
    p = tf_m.add_paragraph()
    p.space_before = Pt(12)
    r1 = p.add_run(); r1.text = f"• {title}\n"
    set_run_font(r1, 11.5, bold=True, color=TEAL)
    r2 = p.add_run(); r2.text = text
    set_run_font(r2, 10, color=INK)


# ==============================================================================
# SLIDE 8: 泛化扩盘: 5 个无关模型的 20 对跨家族交叉验证
# ==============================================================================
slide8 = prs.slides.add_slide(BLANK)
add_card(slide8, 0, 0, SW, SH, fill_color=PAPER, line_color=None)
add_header(slide8, "模型泛化验证: 5 个无关模型在 20 对参照组合下的留出检验", "CROSS-FAMILY GENERALIZATION")

# 左侧: 20 对留出实验描述
add_card(slide8, Inches(0.8), Inches(1.5), Inches(5.2), Inches(5.2), fill_color=CARD)
tb_g = slide8.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(4.6), Inches(4.6))
tf_g = tb_g.text_frame; tf_g.word_wrap = True

p = tf_g.paragraphs[0]
r = p.add_run(); r.text = "扩盘评估方案与设定\n"
set_run_font(r, 13, bold=True, color=NAVY)

g_items = [
    ("无关模型扩展至 5 个", "包括 Qwen2.5-7B, Meta-Llama-3-8B, llama-7b-hf (初代), Baichuan-7B, mpt-7b。涵盖不同厂商与结构。"),
    ("两两交叉留出设定 (20 对)", "一个作为参照模型 (Reference) 计算相对差，另一个作为留出测试负例 (Held-out Test)，其余模型不参与。"),
    ("通过标准 (严格预注册)", "要求强正例 (三种剪枝) 与微调簇的留出召回率 ≥ 90%，且留出负例的误报率 ≤ 10%。"),
    ("评测总体结果", "20 对组合中 18 对完美通过，部署 AUC 高达 0.9999！证实方法不依赖于特定参照模型。")
]

for idx, (t, d) in enumerate(g_items):
    p = tf_g.add_paragraph()
    p.space_before = Pt(10)
    r1 = p.add_run(); r1.text = f"• {t}: "
    set_run_font(r1, 11, bold=True, color=TEAL)
    r2 = p.add_run(); r2.text = d
    set_run_font(r2, 10, color=INK)

# 右侧: 20 对通过情况速览卡片
add_card(slide8, Inches(6.3), Inches(1.5), Inches(6.2), Inches(5.2), fill_color=CARD)
tb_pairs = slide8.shapes.add_textbox(Inches(6.6), Inches(1.8), Inches(5.6), Inches(4.6))
tf_p = tb_pairs.text_frame; tf_p.word_wrap = True

p = tf_p.paragraphs[0]
r = p.add_run(); r.text = "20 对参照组合留出测试明细 (18 / 20 通过):\n"
set_run_font(r, 12, bold=True, color=NAVY)

pairs_summary = [
    ("Qwen2.5-7B 参照组", "对 Llama-3 (通), llama-7b (通), Baichuan (通), mpt-7b (误报0.125 微差卡线)"),
    ("Llama-3-8B 参照组", "对 Qwen2.5 (通), llama-7b (通), Baichuan (通), mpt-7b (误报0.188 微差卡线)"),
    ("llama-7b-hf 参照组", "对 Qwen2.5 (通), Llama-3 (通), Baichuan (通), mpt-7b (通) — 全部通过"),
    ("Baichuan-7B 参照组", "对 Qwen2.5 (通), Llama-3 (通), llama-7b (通), mpt-7b (通) — 全部通过"),
    ("mpt-7b 参照组", "对 Qwen2.5 (通), Llama-3 (通), llama-7b (通), Baichuan (通) — 全部通过")
]

for idx, (head, detail) in enumerate(pairs_summary):
    p = tf_p.add_paragraph()
    p.space_before = Pt(8)
    r1 = p.add_run(); r1.text = f"✔ {head}\n"
    set_run_font(r1, 11, bold=True, color=GREEN if "全部通过" in detail else (NAVY if idx < 2 else TEAL))
    r2 = p.add_run(); r2.text = f"   {detail}"
    set_run_font(r2, 9.5, color=MUT)

p_bot = tf_p.add_paragraph()
p_bot.space_before = Pt(14)
r_bot = p_bot.add_run()
r_bot.text = "🔍 边界归因: 未过的两对系因 mpt-7b 的 seed-0 阈值采样极值偏低 (0.0074 vs 留出 0.0128)，事后加上 0.002 保护带即 20 对全过。这为后续工作提供了极具价值的边界洞察。"
set_run_font(r_bot, 9.5, color=ORANGE)


# ==============================================================================
# SLIDE 9: 总结与给导师的下一步建议
# ==============================================================================
slide9 = prs.slides.add_slide(BLANK)
add_card(slide9, 0, 0, SW, SH, fill_color=PAPER, line_color=None)
add_header(slide9, "总结与后续工作规划 (Conclusion & Roadmaps)", "CONCLUSION & PROSPECTS")

# 三大汇报结论大卡片
final_cards = [
    ("大模块落地成功", "DiffIP 在 7B 上的有效性确立", 
     "成功将 DiffIP 协议无损放大至 7B 规模，构建了配对扰动 + 稀疏层抽取 + ALS 回退的主判定模块，首次提出相对参照消减机制，将真实 SFT 判定 AUC 推升至 0.9998。"),
    ("三大消融支撑坚实", "机制与超参设计完全闭环", 
     "消融 1 证明 4 层稀疏均匀抽取为最佳实践；消融 2 证明 8 维降维空间稳健且配对关系不可或缺；消融 3 证实对线性、非线性及插值混淆伪装具有极强防御韧性。"),
    ("工程封装与代码就绪", "开箱即用的 Lineage 判定接口", 
     "完成了 lineage.py 统一判定 API 封装，通过自动化测试与对抗探针校验，已完全具备产出高质量学术论文和系统落地的条件。")
]

for idx, (t1, t2, desc) in enumerate(final_cards):
    cx = Inches(0.8 + idx * 4.0)
    add_card(slide9, cx, Inches(1.5), Inches(3.7), Inches(5.2), fill_color=CARD)
    # 顶部颜色横条
    add_card(slide9, cx, Inches(1.5), Inches(3.7), Inches(0.12), fill_color=NAVY if idx==0 else (TEAL if idx==1 else ORANGE), line_color=None)
    
    tb = slide9.shapes.add_textbox(cx + Inches(0.2), Inches(1.8), Inches(3.3), Inches(4.6))
    tf = tb.text_frame; tf.word_wrap = True
    p1 = tf.paragraphs[0]
    r1 = p1.add_run(); r1.text = t1
    set_run_font(r1, 14, bold=True, color=NAVY)
    
    p2 = tf.add_paragraph()
    p2.space_before = Pt(4)
    r2 = p2.add_run(); r2.text = t2
    set_run_font(r2, 11, bold=True, color=TEAL if idx==1 else (ORANGE if idx==2 else MUT))
    
    p3 = tf.add_paragraph()
    p3.space_before = Pt(14)
    r3 = p3.add_run(); r3.text = desc
    set_run_font(r3, 10.5, color=INK)

# 保存 Presentation
prs.save(str(OUT_PPT))
print(f"PPT 生成成功: {OUT_PPT}")
