# -*- coding: utf-8 -*-
import sys
import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Academic Palette (Hainan University Style)
    C_BLUE = RGBColor(0, 92, 168)          # #005CA8 主色海大蓝
    C_DARK_BLUE = RGBColor(11, 60, 104)    # #0B3C68 墨蓝
    C_LIGHT_BLUE = RGBColor(238, 245, 252) # #EEF5FC 浅蓝底色
    C_TEXT = RGBColor(34, 34, 34)          # #222222 正文字炭黑
    C_MUTED = RGBColor(95, 99, 104)        # #5F6368 次要文字深灰
    C_BG_CARD = RGBColor(250, 251, 252)    # #FAFBFD 卡片背景
    C_BORDER = RGBColor(218, 224, 233)     # #DAE0E9 边框淡蓝灰
    C_WHITE = RGBColor(255, 255, 255)      # 白色
    C_RED = RGBColor(192, 57, 43)          # #C0392B 警示砖红
    C_GREEN = RGBColor(39, 140, 70)        # #278C46 墨绿
    C_GOLD = RGBColor(211, 84, 0)          # #D35400 琥珀金

    logo_path = r"d:\study\Diffip\llm_migration_v4\hainanu_logo.png"

    def add_header(slide, title, tracker, slide_num, total_slides=16):
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.88), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = C_BORDER
        line.line.color.rgb = C_BORDER

        # Tracker
        tx_top = slide.shapes.add_textbox(Inches(0.8), Inches(0.24), Inches(8.5), Inches(0.28))
        tf_top = tx_top.text_frame
        tf_top.word_wrap = True
        tf_top.margin_left = tf_top.margin_right = tf_top.margin_top = tf_top.margin_bottom = 0
        p_top = tf_top.paragraphs[0]
        p_top.text = f"DiffIP 迁移大模型 · 实验汇报 (v4)   |   {tracker}"
        p_top.font.name = "Microsoft YaHei"
        p_top.font.size = Pt(10)
        p_top.font.bold = True
        p_top.font.color.rgb = C_BLUE

        # Slide Number
        tx_num = slide.shapes.add_textbox(Inches(10.8), Inches(0.24), Inches(1.1), Inches(0.28))
        tf_num = tx_num.text_frame
        tf_num.word_wrap = False
        tf_num.margin_left = tf_num.margin_right = tf_num.margin_top = tf_num.margin_bottom = 0
        p_num = tf_num.paragraphs[0]
        p_num.text = f"{slide_num} / {total_slides}"
        p_num.alignment = PP_ALIGN.RIGHT
        p_num.font.name = "Calibri"
        p_num.font.size = Pt(11)
        p_num.font.bold = True
        p_num.font.color.rgb = C_MUTED

        # Logo
        if os.path.exists(logo_path):
            slide.shapes.add_picture(logo_path, Inches(12.05), Inches(0.18), Inches(0.48), Inches(0.54))

        # Title
        tx_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.48), Inches(10.5), Inches(0.38))
        tf_title = tx_title.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.name = "Microsoft YaHei"
        p_title.font.size = Pt(15.5)
        p_title.font.bold = True
        p_title.font.color.rgb = C_DARK_BLUE

        # Footer
        tx_bot = slide.shapes.add_textbox(Inches(0.8), Inches(7.16), Inches(11.733), Inches(0.22))
        tf_bot = tx_bot.text_frame
        tf_bot.margin_left = tf_bot.margin_right = tf_bot.margin_top = tf_bot.margin_bottom = 0
        p_bot = tf_bot.paragraphs[0]
        p_bot.text = "海南大学计算机科学与技术学院 · 大模型安全课题组 · 2026-11-08"
        p_bot.font.name = "Microsoft YaHei"
        p_bot.font.size = Pt(8.5)
        p_bot.font.color.rgb = C_MUTED

    def add_card(slide, left, top, width, height, title=None, title_color=C_BLUE, bg_color=C_BG_CARD, border_color=C_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.14), width - Inches(0.4), Inches(0.32))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = title
            p.font.name = "Microsoft YaHei"
            p.font.size = Pt(12.5)
            p.font.bold = True
            p.font.color.rgb = title_color

            tb_body = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.48), width - Inches(0.4), height - Inches(0.6))
            tf_body = tb_body.text_frame
            tf_body.word_wrap = True
            tf_body.margin_left = tf_body.margin_right = tf_body.margin_top = tf_body.margin_bottom = 0
            return card, tf_body

        tb_body = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.18), width - Inches(0.4), height - Inches(0.36))
        tf_body = tb_body.text_frame
        tf_body.word_wrap = True
        tf_body.margin_left = tf_body.margin_right = tf_body.margin_top = tf_body.margin_bottom = 0
        return card, tf_body

    def add_bullet(tf, text, prefix="• ", bold_prefix=True, pt_size=10.5, text_color=C_TEXT, space_before=4):
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.space_before = Pt(space_before)
        p.space_after = Pt(2)
        
        if bold_prefix and prefix:
            r1 = p.add_run()
            r1.text = prefix
            r1.font.name = "Microsoft YaHei"
            r1.font.size = Pt(pt_size)
            r1.font.bold = True
            r1.font.color.rgb = C_BLUE
        elif prefix:
            text = prefix + text

        if "【" in text and "】" in text:
            parts = text.split("【")
            if parts[0]:
                r = p.add_run()
                r.text = parts[0]
                r.font.name = "Microsoft YaHei"
                r.font.size = Pt(pt_size)
                r.font.color.rgb = text_color
            for sub in parts[1:]:
                kw, rest = sub.split("】", 1)
                r_kw = p.add_run()
                r_kw.text = f"【{kw}】"
                r_kw.font.name = "Microsoft YaHei"
                r_kw.font.size = Pt(pt_size)
                r_kw.font.bold = True
                r_kw.font.color.rgb = C_BLUE
                if rest:
                    r_rest = p.add_run()
                    r_rest.text = rest
                    r_rest.font.name = "Microsoft YaHei"
                    r_rest.font.size = Pt(pt_size)
                    r_rest.font.color.rgb = text_color
        else:
            r = p.add_run()
            r.text = text
            r.font.name = "Microsoft YaHei"
            r.font.size = Pt(pt_size)
            r.font.color.rgb = text_color

    def create_table(slide, left, top, width, height, rows, cols, headers, data, col_widths=None):
        table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
        table = table_shape.table
        if col_widths:
            for idx, w in enumerate(col_widths):
                table.columns[idx].width = w

        for c_idx, h_text in enumerate(headers):
            cell = table.cell(0, c_idx)
            cell.text = h_text
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_LIGHT_BLUE
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                p.font.name = "Microsoft YaHei"
                p.font.size = Pt(9.5)
                p.font.bold = True
                p.font.color.rgb = C_DARK_BLUE

        for r_idx, row_vals in enumerate(data):
            for c_idx, val in enumerate(row_vals):
                cell = table.cell(r_idx + 1, c_idx)
                cell.text = str(val)
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_WHITE if r_idx % 2 == 0 else C_BG_CARD
                for p in cell.text_frame.paragraphs:
                    p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
                    p.font.name = "Microsoft YaHei"
                    p.font.size = Pt(9.0)
                    p.font.color.rgb = C_TEXT
                    if "通过" in str(val) or "1.00" in str(val):
                        p.font.bold = True
                    if "否" in str(val):
                        p.font.bold = True
                        p.font.color.rgb = C_RED

        return table_shape

    # ==========================================
    # SLIDE 1: 封面
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    bar1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.12), Inches(3.6))
    bar1.fill.solid()
    bar1.fill.fore_color.rgb = C_BLUE
    bar1.line.fill.background()

    if os.path.exists(logo_path):
        s1.shapes.add_picture(logo_path, Inches(11.2), Inches(0.8), Inches(1.2), Inches(1.35))

    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.0), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "DiffIP 迁移到大语言模型"
    p1.font.name = "Microsoft YaHei"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = C_BLUE

    p2 = tf.add_paragraph()
    p2.text = "7B 规模验证、相对判定协议与系统消融实验汇报"
    p2.font.name = "Microsoft YaHei"
    p2.font.size = Pt(20)
    p2.font.bold = True
    p2.font.color.rgb = C_DARK_BLUE
    p2.space_before = Pt(10)

    p3 = tf.add_paragraph()
    p3.text = "面向导师阶段性工作汇报 · 对应前次 v3 汇报之升级演进"
    p3.font.name = "Microsoft YaHei"
    p3.font.size = Pt(13)
    p3.font.color.rgb = C_MUTED
    p3.space_before = Pt(12)

    c_info = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(4.3), Inches(9.5), Inches(1.2))
    c_info.fill.solid()
    c_info.fill.fore_color.rgb = C_LIGHT_BLUE
    c_info.line.color.rgb = C_BORDER
    tb_info = s1.shapes.add_textbox(Inches(1.4), Inches(4.45), Inches(9.1), Inches(0.9))
    tf_i = tb_info.text_frame
    tf_i.word_wrap = True
    p_i1 = tf_i.paragraphs[0]
    p_i1.text = "汇报人：陆顺鑫     |     指导导师：导师     |     研究方向：大模型安全（版权保护与模型指纹）"
    p_i1.font.name = "Microsoft YaHei"
    p_i1.font.size = Pt(12)
    p_i1.font.bold = True
    p_i1.font.color.rgb = C_TEXT

    p_i2 = tf_i.add_paragraph()
    p_i2.text = "培养单位：海南大学计算机科学与技术学院     |     汇报日期：2026 年 11 月 08 日"
    p_i2.font.name = "Microsoft YaHei"
    p_i2.font.size = Pt(11)
    p_i2.font.color.rgb = C_MUTED
    p_i2.space_before = Pt(6)

    # ==========================================
    # SLIDE 2: 进展概览与对齐导师要求
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "问题与进展概览 · 承接 v3 遗留问题与对齐导师要求", "背景与总体规划", 2)

    _, tf_c1 = add_card(s2, Inches(0.8), Inches(1.08), Inches(3.7), Inches(5.85), "前次 v3（0.5B）遗留问题", C_BLUE)
    add_bullet(tf_c1, "【受害模型规模受限】前次验证停留在 Qwen2.5-0.5B 与 SmolLM2-135M，负例面板偏弱，尚未在 7B 主力模型检验。", prefix="◆ ")
    add_bullet(tf_c1, "【缺乏结构变体维度】仅测试了 SFT 微调和人工权重扰动，缺少真实的同底座网络剪枝等高价值结构变体。", prefix="◆ ")
    add_bullet(tf_c1, "【绝对阈值家族依赖】v3 发现绝对距离尺度随模型家族漂移，无法设定横跨所有模型的全局单一阈值。", prefix="◆ ")
    add_bullet(tf_c1, "【消融体系未完备】维数、层数抽法、扰动样本数尚缺乏在真实大模型上的系统化对照论证。", prefix="◆ ")

    _, tf_c2 = add_card(s2, Inches(4.8), Inches(1.08), Inches(3.7), Inches(5.85), "本次 v4 对齐导师要求", C_DARK_BLUE)
    add_bullet(tf_c2, "【一个大模块：7B 迁移与判定协议】受害模型全面升级至 Llama-2-7B，提出破除尺度漂移的『同一 Prompt 相对判定协议』，并在 5 模型 20 对跨家族参照中严谨验证。", prefix="◆ ")
    add_bullet(tf_c2, "【消融 1：降维维数选择】固定 4 层在 8、10、15、20、30、40 维之间严格根据预注册规则比选。", prefix="◆ ")
    add_bullet(tf_c2, "【消融 2：层数与抽取方式】对比均匀 4 层、均匀 8 层、全部 32 层与末端 4 层，打破『层数越多越好』的误区。", prefix="◆ ")
    add_bullet(tf_c2, "【消融 3：扰动样本数收敛】对比原文 N=20 与 N=128，揭示样本量对负例虚高分数的抑制机理。", prefix="◆ ")

    _, tf_c3 = add_card(s2, Inches(8.8), Inches(1.08), Inches(3.7), Inches(5.85), "核心结论与实事求是态度", C_GREEN)
    add_bullet(tf_c3, "【谱系排序高度稳固】7B 上谱系排序完整保持：剪枝 ≫ 微调及线性伪装 > 中文衍生 > 无关模型，线性伪装完全隐形。", prefix="◆ ")
    add_bullet(tf_c3, "【相对判定全面达标】留出召回达 0.97~1.00，误报 0~0.03，部署 AUC 高达 0.9998。", prefix="◆ ")
    add_bullet(tf_c3, "【客观呈现阴性结果】黑盒蒸馏学生无法检出（负结果，AUC 0.493），深改中文衍生版处于检测临界区（召回 ~0.50）。", prefix="◆ ")
    add_bullet(tf_c3, "【恪守学术预注册规范】20 对参照过 18 对，实事求是分析未过 2 对的阈值尾部波动，不拿事后规则掩饰。", prefix="◆ ")

    # ==========================================
    # SLIDE 3: 方法设计与 LLM 迁移协议
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "方法设计 · 扩散模型到大语言模型的映射与协议", "核心方法与迁移机制", 3)

    _, tf_m1 = add_card(s3, Inches(0.8), Inches(1.08), Inches(5.7), Inches(5.85), "DiffIP 原理与 LLM 映射实现", C_BLUE)
    add_bullet(tf_m1, "【扩散模型原生机制】固定 N 个生成种子，取出去噪步特征做 PCA 降维，利用 ALS（正交旋转 Q、缩放 S、平移 h）回拉对齐，以残差指数计算相似度：", prefix="◆ ")
    add_bullet(tf_m1, "sim = exp(-d_avg)，其中 d_avg 为各层 ALS 拟合残差平均值。", prefix="   ", text_color=C_DARK_BLUE)
    add_bullet(tf_m1, "【映射 1：生成种子 → 成对嵌入扰动】大模型无扩散去噪步，改为在输入 Embedding 上叠加严格配对的高斯噪声，标准扰动幅度固定为 σ = 2e-3。", prefix="◆ ")
    add_bullet(tf_m1, "【映射 2：去噪步 → 稀疏关键层表征】将扩散时间步映射为 Transformer 的层级特征。对单条 Prompt 采集 1 份干净输入 + 127 份配对噪声，共 128 个样本，取 token 均值池化向量。", prefix="◆ ")
    add_bullet(tf_m1, "【映射 3：关闭 DTW 规整】LLM 层间对应语义严格，扩散模型中的动态时间规整（DTW）在 LLM 迁移中关闭，避免对层强行拟合产生过拟合伪影。", prefix="◆ ")

    _, tf_m2 = add_card(s3, Inches(6.8), Inches(1.08), Inches(5.733), Inches(5.85), "关键决策：为什么加噪声？为什么不用解码？", C_DARK_BLUE)
    add_bullet(tf_m2, "【自回归解码的『走散效应』】大语言模型逐 token 自回归生成时，一旦在某一步两模型选中不同 token，后续整段序列上下文将彻底走散，样本配对关系瞬间瓦解。", prefix="◆ ")
    add_bullet(tf_m2, "【v3 实测教训：解码采样全面失效】在前次小模型对比中，解码采样相似度全线跌至 0 附近，且无关模型分数甚至倒挂高于微调模型，完全无法判别谱系。", prefix="◆ ")
    add_bullet(tf_m2, "【成对噪声：真正的差分测量】固定两侧生成随机数种子，输入相同顺序的同一串噪声。抵消提示词本身的语义背景，只测量两模型对微小局部扰动的几何响应差异。", prefix="◆ ")
    add_bullet(tf_m2, "【严禁异质提示词混成一朵云】必须单条 Prompt 内部生成配对云打分，不能将中、英、代码等不同类型混在一次对齐中；多峰语义空间会导致单组正交变换彻底失效。", prefix="◆ ")

    # ==========================================
    # SLIDE 4: 7B 核心发现：绝对相似度失效
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "7B 核心发现 · 绝对相似度失效的原因深度剖析", "大模块 · 核心突破之动因", 4)

    _, tf_p1 = add_card(s4, Inches(0.8), Inches(1.08), Inches(5.7), Inches(5.85), "现象：绝对裕度收窄与严重重叠", C_BLUE)
    add_bullet(tf_p1, "【0.5B 到 7B 的分值漂移】0.5B 上官方微调均值 0.760 vs 负例 0.238（裕度达 0.522）；而 7B 上官方微调均值降至 0.363，负例均值 0.254，平均绝对裕度收窄至仅 0.109。", prefix="◆ ")
    add_bullet(tf_p1, "【提示词内巨大方差波动】在 32 条 Prompt 上，官方微调的标准差高达 0.249（最低 0.010，最高 0.853）；两个无关模型的标准差同样达 0.249（最高竟达 0.814）！", prefix="◆ ")
    add_bullet(tf_p1, "【单一全局阈值彻底失效】若取负例误报为 0 的全局硬阈值（0.786），官方微调留出召回仅剩极其惨烈的 0.0625（6.25%），全局 ROC 的 AUC 仅为 0.702。", prefix="◆ ")
    add_bullet(tf_p1, "【跨规模经验无法迁移】若将 0.5B 的中点阈值 0.499 直接套用到 7B，仅能覆盖 32% 的微调样本，同时产生 23% 的负例误报率，绝对阈值不可跨规模移植。", prefix="◆ ")

    _, tf_p2 = add_card(s4, Inches(6.8), Inches(1.08), Inches(5.733), Inches(5.85), "机理揭秘：同高同低之谜（逐 Prompt 强相关）", C_DARK_BLUE)
    add_bullet(tf_p2, "【逐 Prompt 惊人共变：r = 0.956】官方微调与 Qwen2.5-7B、Llama-3-8B 在逐条 Prompt 的均值相关系数高达 0.956！同一条 Prompt 上，正例与负例同步升高或降低。", prefix="◆ ")
    add_bullet(tf_p2, "【Prompt 难度主导，非字数所致】字数长度与相似度相关仅 0.022。去除英文/中文/代码/数学/问答五类均值后，类内偏相关仍高达 0.949。", prefix="◆ ")
    add_bullet(tf_p2, "【任务语义空间差异巨大】数学类 Prompt 的微调均值达 0.533，而问答类仅 0.077。在难提示词上，微调模型的分数甚至远低于简单提示词上的无关模型！", prefix="◆ ")
    add_bullet(tf_p2, "【破局根本线索】绝不能把所有 Prompt 混在一起画一条水平阈值；判定的基准必须落在『同一条 Prompt』的相对差异上！", prefix="◆ ", text_color=C_GREEN)

    # ==========================================
    # SLIDE 5: 核心突破：同一 Prompt 相对判定协议
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "核心突破 · 同一 Prompt 下的相对判定协议", "大模块 · 核心判定体系", 5)

    _, tf_proto = add_card(s5, Inches(0.8), Inches(1.08), Inches(11.733), Inches(2.35), "相对判定协议设计与严谨留出机制", C_BLUE)
    add_bullet(tf_proto, "【判定公式设计】以受害模型为基准，在『同一条 Prompt』下计算待测模型相似度与无关参照模型（Reference）相似度的差值：", prefix="◆ ")
    add_bullet(tf_proto, "Score(M, Prompt) = sim(M, Victim | Prompt) - sim(Ref_neg, Victim | Prompt)", prefix="   ", text_color=C_DARK_BLUE)
    add_bullet(tf_proto, "【双盲留出验证机制】以 Seed 0（第一组随机种子）上留出无关模型的最大间隔确定判定阈值（阈值极低，仅约 0.014 ~ 0.016）；", prefix="◆ ")
    add_bullet(tf_proto, "【严格通过准则】在完全独立的留出集（Seed 1 与 Seed 2）上检验：要求部署微调簇与强正例（剪枝）留出召回率 ≥ 0.90，留出误报率 ≤ 0.10。", prefix="◆ ")

    _, tf_res = add_card(s5, Inches(0.8), Inches(3.55), Inches(5.7), Inches(3.38), "相对判定验证结果（Qwen ↔ Llama-3 互为参照）", C_GREEN)
    add_bullet(tf_res, "【剪枝留出召回】SparseGPT、GBLM、Wanda 三种剪枝在两方向留出召回率均为 1.00 / 1.00，完美通过！", prefix="◆ ")
    add_bullet(tf_res, "【微调留出召回】官方 SFT 及三种线性变换留出召回达 0.969 ~ 1.00，留出误报率仅 0 ~ 0.031。", prefix="◆ ")
    add_bullet(tf_res, "【AUC 飞跃】部署正类对留出无关模型的 AUC 从绝对阈值的 0.702 骤增至惊人的 0.9998！", prefix="◆ ")

    _, tf_ab = add_card(s5, Inches(6.8), Inches(3.55), Inches(5.733), Inches(3.38), "对照消融：证明『同一条 Prompt』是决定性命脉", C_RED)
    add_bullet(tf_ab, "【消融 1：减错 Prompt（错位减）】若用随机不同 Prompt 的无关模型相减，微调留出召回瞬间暴跌至 0.06 ~ 0.11，AUC 跌回 0.644，判定彻底失效！", prefix="◆ ")
    add_bullet(tf_ab, "【消融 2：减无关模型全局均值】若仅减去负例在 32 条 Prompt 上的全局平均值，微调召回仅 0.06，AUC 为 0.702，退化回绝对阈值状态！", prefix="◆ ")
    add_bullet(tf_ab, "【本质结论】起作用的绝非模型的整体平均位移，而是同一 Prompt 下微观几何曲率的精准差分！", prefix="◆ ", text_color=C_DARK_BLUE)

    # ==========================================
    # SLIDE 6: 7B 主线结果：完整谱系排序
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "7B 主线结果 · 完整谱系排序在真实规模下严格保持", "大模块 · 核心主实验数据", 6)

    headers = ["模型类别", "具体被测模型", "平均相似度", "谱系地位与判定"]
    data = [
        ["受害自身", "Llama-2-7B (Base)", "1.000", "基准 (完全自洽)"],
        ["同底座剪枝", "SparseGPT 剪枝", "0.559", "强正例 (召回 1.00)"],
        ["同底座剪枝", "GBLM 剪枝", "0.538", "强正例 (召回 1.00)"],
        ["同底座剪枝", "Wanda 剪枝", "0.528", "强正例 (召回 1.00)"],
        ["官方微调", "Llama-2-7b-chat-hf (SFT)", "0.363", "部署正类 (召回 >0.97)"],
        ["线性伪装", "权重置换 / 表示置换", "0.364 / 0.363", "伪装隐形 (同 SFT)"],
        ["线性伪装", "各向异性缩放", "0.365", "伪装隐形 (同 SFT)"],
        ["复合微调", "对话版 + 层剪枝", "0.357", "部署正类 (稳定检出)"],
        ["独立大改", "Chinese-llama-2-7b", "0.311", "临界区 (召回 ~0.50)"],
        ["跨家族负例", "Qwen2.5-7B / Llama-3-8B", "0.254 / 0.254", "负例底线 (误报 ≤0.03)"]
    ]
    col_w = [Inches(1.3), Inches(2.2), Inches(1.1), Inches(1.8)]
    create_table(s6, Inches(0.8), Inches(1.15), Inches(6.4), Inches(5.75), 11, 4, headers, data, col_w)

    _, tf_ded = add_card(s6, Inches(7.4), Inches(1.15), Inches(5.133), Inches(5.75), "7B 主实验学术洞察与结论", C_BLUE)
    add_bullet(tf_ded, "【1. 谱系阶梯高度分明】", prefix="", bold_prefix=False, pt_size=11, text_color=C_DARK_BLUE)
    add_bullet(tf_ded, "受害基准 (1.00) ≫ 剪枝 (0.53~0.56) > 微调及伪装 (0.357~0.365) > 独立中文衍生 (0.311) > 跨家族负例 (0.254)。", prefix="  • ")
    add_bullet(tf_ded, "【2. 稀疏剪枝保留最强几何指纹】", prefix="", bold_prefix=False, pt_size=11, text_color=C_DARK_BLUE)
    add_bullet(tf_ded, "同底座剪枝不改变层级拓扑，仅稀疏化权重，其表征残差极小，相似度达 0.53~0.56，是最易识别的同源衍生形式。", prefix="  • ")
    add_bullet(tf_ded, "【3. 三种白盒线性伪装完全隐形】", prefix="", bold_prefix=False, pt_size=11, text_color=C_DARK_BLUE)
    add_bullet(tf_ded, "正交置换与尺度缩放均属于 ALS 算法原生回拉的目标空间，分数与官方微调在噪声范围内完全重合，无法遮蔽谱系！", prefix="  • ")
    add_bullet(tf_ded, "【4. 跨家族负例地板极其稳固】", prefix="", bold_prefix=False, pt_size=11, text_color=C_DARK_BLUE)
    add_bullet(tf_ded, "Qwen 与 Llama-3 均稳定落在 0.254 地板线，为相对差分判定提供了极佳且一致的零点参照基准。", prefix="  • ")

    # ==========================================
    # SLIDE 7: 边界与阴性结果分析
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "边界与阴性结果 · 蒸馏与深度衍生版的实事求是分析", "大模块 · 科学边界剖析", 7)

    _, tf_neg1 = add_card(s7, Inches(0.8), Inches(1.08), Inches(5.7), Inches(5.85), "阴性结果：序列级黑盒蒸馏无法检出", C_RED)
    add_bullet(tf_neg1, "【实验配置】以 Llama-3.2-1B 为教师，训练两个同基座、同计算预算的 1B 学生：一个用教师贪心生成的 Alpaca 文本微调（distill），另一个用原始 Alpaca 文本微调（control）。", prefix="◆ ")
    add_bullet(tf_neg1, "【实验数据】在 8 维、均匀 4 层下：distill 相似度均值为 0.619，control 相似度均值为 0.616。两者差值仅约 0.003，落在一个标准差之内！", prefix="◆ ")
    add_bullet(tf_neg1, "【判定失效】蒸馏学生对对照组的判定 AUC 仅为 0.493（等于完全随机猜测）。", prefix="◆ ")
    add_bullet(tf_neg1, "【深入理论归因】序列级 KD 仅传递生成 token 文本，受害模型的参数流形与内部表征几何差异在文本采样中彻底丢失，微调后呈现的仅仅是『同基座 Alpaca 微调』的共性特征。DiffIP 无法探测纯黑盒文本蒸馏窃取！", prefix="◆ ", text_color=C_DARK_BLUE)

    _, tf_neg2 = add_card(s7, Inches(6.8), Inches(1.08), Inches(5.733), Inches(5.85), "临界状态：深度二次预训练版（Chinese-llama）", C_GOLD)
    add_bullet(tf_neg2, "【模型特性】Chinese-llama-2-7b 是在大规模中文语料上进行了词表扩充与深度二次预训练的重度衍生模型。", prefix="◆ ")
    add_bullet(tf_neg2, "【实验表现】平均相似度为 0.311，显著高于负例地板（0.254）。相对判定中间间隔为 0.019，而低误报安全阈值约为 0.016。", prefix="◆ ")
    add_bullet(tf_neg2, "【留出召回落入临界区】在严格的留出种子检验下，其召回率仅为 0.469 ~ 0.531，未达到事先设定的 ≥ 0.90 高召回通过线。", prefix="◆ ")
    add_bullet(tf_neg2, "【明确学术结论】海量二次预训练带来的参数剧烈漂移改变了局部几何曲率，模型处于可分与不可分的狭窄分水岭，不能与轻度微调共用同一条高灵敏度判定线。", prefix="◆ ", text_color=C_TEXT)

    # ==========================================
    # SLIDE 8: 无关模型盘扩展测试（20 对检验）
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "无关模型盘扩展 · 5 模型 20 对参照的极限压力测试", "大模块 · 泛化性压力测试", 8)

    _, tf_w1 = add_card(s8, Inches(0.8), Inches(1.08), Inches(5.7), Inches(5.85), "测试协议与 18/20 通过率", C_BLUE)
    add_bullet(tf_w1, "【扩展负例库（5 大 7B 模型）】将跨家族参照从 2 个扩充至 5 个：Qwen2.5-7B、Llama-3-8B、llama-7b-hf（一代原版）、Baichuan-7B、mpt-7b。", prefix="◆ ")
    add_bullet(tf_w1, "【20 对两两有序检验】任选一个作为『减法参照』，另一个作为『留出负例』，遍历全部 5×4 = 20 对有序组合。目的是验证相对判定不仅依赖 Qwen，对任意参照模型皆稳固。", prefix="◆ ")
    add_bullet(tf_w1, "【核心结论：18 对完全通过】在严格事先规则下，20 对中 18 对顺利达标，部署正类 AUC 高达惊人的 0.9999！", prefix="◆ ", text_color=C_GREEN)
    add_bullet(tf_w1, "【Qwen ↔ Llama-3 互验】阈值 0.0161，留出微调召回 1.00 / 0.969，误报 0 / 0，证明口径高度统一自洽。", prefix="◆ ")

    _, tf_w2 = add_card(s8, Inches(6.8), Inches(1.08), Inches(5.733), Inches(5.85), "客观复盘未通过的 2 对：误报而非漏检", C_DARK_BLUE)
    add_bullet(tf_w2, "【未过的 2 对详情】均为以 mpt-7b 作为留出负例：Qwen → mpt-7b 与 Llama-3 → mpt-7b。强正例与微调召回均为 1.00，但种子 2 误报率为 0.125 与 0.188，略超 0.10 警戒线。", prefix="◆ ")
    add_bullet(tf_w2, "【根因排查：非模型相似，系阈值尾部抖动】mpt-7b 的均值是 0.2528，其余负例是 0.2525~0.2537，逐 Prompt 相关 >0.99994。两模型在几何上完全等价！", prefix="◆ ")
    add_bullet(tf_w2, "【阈值估计的统计弱点】以 32 条 Prompt 的『最大值』作为阈值，在留出种子的极端尾部存在轻微方差抖动（留出最大值微超标定最大值 0.0018）。", prefix="◆ ")
    add_bullet(tf_w2, "【事后验证与整改】事后若增加 0.002 的安全保护带，20 对可 100% 全部通过。实事求是按预注册规则仍记 18/20，展现严谨治学态度。", prefix="◆ ", text_color=C_GOLD)

    # ==========================================
    # SLIDE 9: 消融实验 1：降维维数选择
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "消融实验 ① · 降维维数选择（8 维至 40 维）", "系统消融 · 核心参数论证", 9)

    h_dim = ["维数 (d)", "是否通过", "判定间隔 (Margin)", "标准差 (Std)", "相对 8 维增益"]
    d_dim = [
        ["8 维 (默认)", "是", "0.092", "0.080", "基准 (预注册基线)"],
        ["10 维", "是", "0.093", "0.079", "+0.001 (微弱)"],
        ["15 维", "是", "0.095", "0.078", "+0.003 (微弱)"],
        ["20 维", "是", "0.092", "0.078", "0.000 (持平)"],
        ["30 维", "是", "0.098", "0.077", "+0.006 (未达门槛)"],
        ["40 维", "是", "0.098", "0.077", "+0.006 (未达门槛)"]
    ]
    w_dim = [Inches(1.3), Inches(1.1), Inches(1.4), Inches(1.3), Inches(1.6)]
    create_table(s9, Inches(0.8), Inches(1.3), Inches(6.7), Inches(4.5), 7, 5, h_dim, d_dim, w_dim)

    _, tf_dim = add_card(s9, Inches(7.7), Inches(1.15), Inches(4.833), Inches(5.75), "维数选择与预注册准则推导", C_BLUE)
    add_bullet(tf_dim, "【事先预注册准则】更换默认 8 维的设计标准：间隔必须至少提升 0.02 且标准差不增加，否则恪守奥卡姆剃刀原则保留 8 维。", prefix="◆ ")
    add_bullet(tf_dim, "【实验结论：全维全通，维持 8 维】六个维数设置在相对判定协议下全部顺利通过；即使最高扩展至 40 维，间隔提升仅有 0.006，远未达到 0.02 门槛，正式记录为 gap_not_clear，坚守 8 维。", prefix="◆ ")
    add_bullet(tf_dim, "【深层几何表征机制】说明输入加噪后的扰动流形主要能量高度集中在极低维正交子空间中（前 8 维已解释绝大部分主成分方差）。", prefix="◆ ")
    add_bullet(tf_dim, "【计算复杂度优势】维持 8 维使 ALS 的矩阵运算维持在极轻量级，完全摆脱了高维计算灾难，工程实用价值极高。", prefix="◆ ", text_color=C_GREEN)

    # ==========================================
    # SLIDE 10: 消融实验 2：层数与抽取方式探索
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "消融实验 ② · 层数与抽取方式的全局探索", "系统消融 · 拓扑设计论证", 10)

    h_layer = ["抽取策略", "具体层数配置", "是否通过", "间隔", "留出微调召回"]
    d_layer = [
        ["均匀 4 层 (默认)", "第 0, 10, 21, 31 层", "是 (峰值)", "0.092", "0.969 ~ 1.00"],
        ["均匀 8 层", "等间隔 8 层", "是", "0.066", "0.938 ~ 0.969"],
        ["全部 32 层", "提取所有 32 层", "否", "0.043", "0.71 ~ 0.79 (严重脱靶)"],
        ["最后 4 层", "第 28, 29, 30, 31 层", "否", "0.033", "0.66 ~ 0.78 (严重脱靶)"],
        ["中间 2 层", "第 10, 21 层", "否", "0.079", "0.79 ~ 0.80 (未达 0.90)"]
    ]
    w_layer = [Inches(1.5), Inches(1.8), Inches(1.1), Inches(0.9), Inches(1.6)]
    create_table(s10, Inches(0.8), Inches(1.2), Inches(6.9), Inches(5.0), 6, 5, h_layer, d_layer, w_layer)

    _, tf_lyr = add_card(s10, Inches(7.9), Inches(1.15), Inches(4.633), Inches(5.75), "『层数非越多越好』的理论洞察", C_BLUE)
    add_bullet(tf_lyr, "【破除直觉偏见】打破了『采集层数越多、信息量越完备、判定越准』的直觉误区，确立了『均匀 4 层即是全局峰值』的关键学术发现。", prefix="◆ ")
    add_bullet(tf_lyr, "【深层局部语义噪声假说】末端高层（如最后 4 层）极度贴近特定任务语义输出，局部神经元激活剧烈扰动，为几何对齐引入了大量高频噪声（间隔骤降至 0.033，召回崩至 0.66）。", prefix="◆ ")
    add_bullet(tf_lyr, "【均匀稀疏采样的优越性】从浅层（通用语法）、中层（上下文组合）到深层（语义抽象）均匀取样，能够天然平滑单层特征畸变，以极低维度构筑最纯净的流形指纹。", prefix="◆ ", text_color=C_GREEN)

    # ==========================================
    # SLIDE 11: 消融实验 3：扰动样本数收敛性
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "消融实验 ③ · 扰动样本数对收敛性与虚高分数的抑制", "系统消融 · 采样预算论证", 11)

    _, tf_sm1 = add_card(s11, Inches(0.8), Inches(1.08), Inches(5.7), Inches(5.85), "原文 N=20 与迁移后 N=128 的对比", C_BLUE)
    add_bullet(tf_sm1, "【扩散模型原生参数 N=20】DiffIP 原文在图像扩散模型中每个 Prompt 仅取 N=20 个种子样本。", prefix="◆ ")
    add_bullet(tf_sm1, "【迁移至 LLM 时的水土不服】在 7B 相对判定协议下测试 N=20：留出微调召回仅 0.78~0.84，无法达到事先设定的 ≥ 0.90 红线，判定宣告失败！", prefix="◆ ")
    add_bullet(tf_sm1, "【扩充至 N=128 的完美表现】样本量扩充至 128 时，相对判定在正负例两端全面稳固通过（召回 0.97~1.00，误报 0~0.03）。", prefix="◆ ")
    add_bullet(tf_sm1, "【明确结论】DiffIP 迁移至离散文本 LLM 时，绝不能机械照搬原论文的 N=20 预算，必须匹配 N=128 充裕配对云。", prefix="◆ ", text_color=C_GREEN)

    _, tf_sm2 = add_card(s11, Inches(6.8), Inches(1.08), Inches(5.733), Inches(5.85), "小样本为何会导致负例分数『虚高』？", C_DARK_BLUE)
    add_bullet(tf_sm2, "【高维空间虚假对齐效应】在小样本（N=8）时，负例相似度被虚假推高至 0.576；N=20 时负例仍高达 0.328；唯有扩充至 N=128 时，负例分数才彻底回落至 0.238 ~ 0.254 的稳态基底。", prefix="◆ ")
    add_bullet(tf_sm2, "【数学机理推导】ALS 具有正交旋转 Q、对角缩放 S 与平移 h 的高度自适应参数。当配对样本云过于稀疏时，算法极易在无关模型之间找到『偶然的局部对齐超平面』，导致残差虚小、相似度虚高。", prefix="◆ ")
    add_bullet(tf_sm2, "【充分充盈流形】128 个扰动样本能够充分张成微观几何邻域，彻底粉碎偶然拟合，为相对判定提供坚实置信度。", prefix="◆ ", text_color=C_TEXT)

    # ==========================================
    # SLIDE 12: 补充机理：配对几何与抗伪装鲁棒性
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "补充机理 · 配对几何的真实性与抗特征伪装鲁棒性", "深度机理验证", 12)

    _, tf_sh1 = add_card(s12, Inches(0.8), Inches(1.08), Inches(5.7), Inches(5.85), "配对打乱（Shuffle）验证：信号唯独来自成对几何", C_BLUE)
    add_bullet(tf_sh1, "【严密检验假说】相似度的高分究竟来自两模型宏观特征分布的相似，还是来自成对扰动下微观几何轨迹的一致？", prefix="◆ ")
    add_bullet(tf_sh1, "【Shuffle 实验设计】将官方微调模型的 128 个扰动样本在送入对齐前完全打乱配对顺序，切断与受害模型的一一对应关系。", prefix="◆ ")
    add_bullet(tf_sh1, "【结果：瞬间跌回负例地板】打乱后相似度均值从 0.363 骤降至 0.253（与无关负例完全重合），32 条 Prompt 里仅胜 5 条！", prefix="◆ ")
    add_bullet(tf_sh1, "【确凿科学论断】确凿证明 DiffIP 提取的指纹 100% 来自同一噪声扰动下的差分几何拓扑，非配对特征毫无防伪信号！", prefix="◆ ", text_color=C_GREEN)

    _, tf_sh2 = add_card(s12, Inches(6.8), Inches(1.08), Inches(5.733), Inches(5.85), "对抗伪装：非线性扭曲与特征掺杂攻击测试", C_DARK_BLUE)
    add_bullet(tf_sh2, "【9 种非线性特征伪装】对微调表征施加包括符号保持平方、正态分位数变换等 9 种非线性破坏：8 种依然落在原生微调置信带内（32 条全胜）；最强破坏的平方变换均值仍达 0.344（胜 31/32 条）。", prefix="◆ ")
    add_bullet(tf_sh2, "【特征混合攻击（Mixup）】按能量拉齐后向微调特征掺入高达 75% 的 Llama-3 无关特征：微调均值降至 0.259（贴近地板），但逐 Prompt 对 Qwen 依然胜 27/32 条！", prefix="◆ ")
    add_bullet(tf_sh2, "【防伪安全性】攻击者单纯在特征层做非线性平滑或混合加噪，根本无法抹除底层的微观配对几何。", prefix="◆ ", text_color=C_TEXT)

    # ==========================================
    # SLIDE 13: 跨规模对比（0.5B vs 7B）
    # ==========================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "跨规模演化 · 0.5B 到 7B 的跨越：什么不变，什么改变", "规模扩展规律总结", 13)

    _, tf_inv = add_card(s13, Inches(0.8), Inches(1.08), Inches(5.7), Inches(5.85), "跨规模保持不变的核心规律（Invariants）", C_BLUE)
    add_bullet(tf_inv, "【谱系序关系严格不变】无论在 0.5B 还是 7B，排序永远保持：剪枝 ≫ 微调及伪装 > 独立中文衍生 > 跨家族负例地板，梯次坚不可摧。", prefix="◆ ")
    add_bullet(tf_inv, "【白盒线性伪装完全透明】权重置换、表示置换与尺度缩放等常规伪装手段在任何模型规模下均被 ALS 完美抵消，完全隐形。", prefix="◆ ")
    add_bullet(tf_inv, "【配对扰动几何依赖】均依赖 N=128 的成对输入扰动测量局部差分流形，单样本或打乱样本均失效。", prefix="◆ ")
    add_bullet(tf_inv, "【8 维降维空间普适】PCA 降维至 8 维始终是捕获主成分与抗过拟合的最优权衡点。", prefix="◆ ")

    _, tf_var = add_card(s13, Inches(6.8), Inches(1.08), Inches(5.733), Inches(5.85), "跨规模发生根本变化的核心机制（Variants）", C_DARK_BLUE)
    add_bullet(tf_var, "【绝对判定裕度大幅收窄】0.5B 上微调与负例相差约 0.52，而在 7B 上绝对均值差距压缩至 0.11，对判别协议精度提出极端要求。", prefix="◆ ")
    add_bullet(tf_var, "【Prompt 内部方差显著放大】7B 模型参数容量巨大，在不同类型任务（如数学 vs 对话）上的表征流形发散度极高，单 Prompt 标准差扩大至 0.25。", prefix="◆ ")
    add_bullet(tf_var, "【判定范式发生跃迁】从 0.5B 尝试设定单模型『绝对阈值』，根本性升级为 7B 的同一 Prompt『相对差分判定』，完成了方法论蜕变。", prefix="◆ ", text_color=C_GREEN)

    # ==========================================
    # SLIDE 14: 局限性诚实记录
    # ==========================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "局限性记录 · 实事求是的学术态度与待攻克难点", "科研反思与局限剖析", 14)

    _, tf_l1 = add_card(s14, Inches(0.8), Inches(1.15), Inches(3.7), Inches(5.75), "局限 1：阈值估计尾部方差", C_BLUE)
    add_bullet(tf_l1, "【现象记录】无关模型 20 对测试中过 18 对，未过的 2 对系 mpt-7b 产生 0.125 和 0.188 误报。", prefix="◆ ")
    add_bullet(tf_l1, "【问题定位】仅用 32 条 Prompt 的『最大值』确定阈值，在极端尾部统计样本不足，容易受极值方差波动影响。", prefix="◆ ")
    add_bullet(tf_l1, "【应对准则】事后加保护带虽可全过，但预注册方案必须坚持事先规定规则，需在后续版本中规范化阈值保护带。", prefix="◆ ")

    _, tf_l2 = add_card(s14, Inches(4.8), Inches(1.15), Inches(3.7), Inches(5.75), "局限 2：缺失 7B 同口径 REEF 对照", C_DARK_BLUE)
    add_bullet(tf_l2, "【现状记录】目前与经典白盒模型指纹 REEF 的严格细粒度对比主要在 0.5B 小模型完成（DiffIP 0.956 vs REEF 0.923）。", prefix="◆ ")
    add_bullet(tf_l2, "【缺口定位】在 7B 主线上，由于计算资源与时间窗口，尚未拉起与 REEF 的全量同口径对比。", prefix="◆ ")
    add_bullet(tf_l2, "【下一步补全】需在同一套 7B 特征与扰动云上，严格跑出 CKA 指标与 AUC，形成同等严格的对照证据链。", prefix="◆ ")

    _, tf_l3 = add_card(s14, Inches(8.8), Inches(1.15), Inches(3.7), Inches(5.75), "局限 3：真实独立衍生模型厚度", C_GREEN)
    add_bullet(tf_l3, "【现状记录】当前 7B 的同谱系测试以官方 SFT 和同底座稀疏剪枝为主，线性伪装模型亦由 SFT 派生。", prefix="◆ ")
    add_bullet(tf_l3, "【拓展空间】本地储备的 Vicuna、WizardMath、CodeLlama 等多领域开源微调模型尚未全部接入主线测试。", prefix="◆ ")
    add_bullet(tf_l3, "【学术价值】接入更多真实开源衍生版本，可进一步夯实方法在复杂专业领域微调下的泛化能力。", prefix="◆ ")

    # ==========================================
    # SLIDE 15: 下一步工作与演进计划
    # ==========================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "下一步工作 · 针对局限性的清晰演进规划", "后续研究推进路线", 15)

    _, tf_pl1 = add_card(s15, Inches(0.8), Inches(1.15), Inches(3.7), Inches(5.75), "演进 1：相对判定协议稳健化", C_BLUE)
    add_bullet(tf_pl1, "【预注册安全保护带】在方案中正式事先加入固定间隔保护带（如 Δ_margin = 0.002）或引入非参数化分位数阈值估计（如 95 分位数代替极值）。", prefix="◆ ")
    add_bullet(tf_pl1, "【扩充校准 Prompt 规模】将阈值标定 Prompt 从 32 条扩充至 64~128 条，彻底平抑种子尾部抖动。", prefix="◆ ")
    add_bullet(tf_pl1, "【新种子独立复验】用全新 Seed 集合全面验证 20 对无关模型 100% 稳定过线。", prefix="◆ ")

    _, tf_pl2 = add_card(s15, Inches(4.8), Inches(1.15), Inches(3.7), Inches(5.75), "演进 2：补齐 7B 上 REEF 对照", C_DARK_BLUE)
    add_bullet(tf_pl2, "【同一特征集评估】直接复用已保存的 7B even4 规范特征，排除采样偶然性影响。", prefix="◆ ")
    add_bullet(tf_pl2, "【对比 CKA 与 DiffIP】计算扰动云上的 CKA 矩阵，评估其在 7B 剪枝与微调模型上的细粒度分类 AUC 与裕度。", prefix="◆ ")
    add_bullet(tf_pl2, "【论文级对照表格】形成对标顶会规范的标准主对比表，明晰 ALS 回拉相比 CKA 的核心几何增益。", prefix="◆ ")

    _, tf_pl3 = add_card(s15, Inches(8.8), Inches(1.15), Inches(3.7), Inches(5.75), "演进 3：扩充真实微调变体库", C_GREEN)
    add_bullet(tf_pl3, "【接入多领域微调版】接入基于 Llama-2-7B 的 Vicuna（多轮对话）、WizardMath（强化推理）、CodeLlama（代码专业版）。", prefix="◆ ")
    add_bullet(tf_pl3, "【探索 LoRA 轻量衍生】验证低秩适配（LoRA）微调在相对判定下的可辨识度与指纹保留规律。", prefix="◆ ")
    add_bullet(tf_pl3, "【量化与混合压缩测试】补充 INT4/INT8 量化以及结构化剪枝的鲁棒性边界。", prefix="◆ ")

    # ==========================================
    # SLIDE 16: 汇报小结与致谢
    # ==========================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "汇报小结 · 敬请导师批评指正", "阶段汇报总结", 16)

    _, tf_sum1 = add_card(s16, Inches(0.8), Inches(1.15), Inches(3.7), Inches(4.5), "大模块成功闭环", C_BLUE)
    add_bullet(tf_sum1, "【完成真实 7B 跃迁】成功实现从扩散模型到 7B 主力语言模型的完整迁移体系。", prefix="◆ ")
    add_bullet(tf_sum1, "【攻克绝对阈值难题】确立『同一 Prompt 相对判定协议』，留出召回率 0.97~1.00，AUC 达 0.9998。", prefix="◆ ")
    add_bullet(tf_sum1, "【20 对压力测试】跨家族参照通过 18/20，展现极强泛化能力。", prefix="◆ ")

    _, tf_sum2 = add_card(s16, Inches(4.8), Inches(1.15), Inches(3.7), Inches(4.5), "消融实验严谨扎实", C_DARK_BLUE)
    add_bullet(tf_sum2, "【维数充分性】6 档维数全通过，维持 8 维低计算负载。", prefix="◆ ")
    add_bullet(tf_sum2, "【层数稀疏峰值】发现均匀 4 层优于全层与深层，证明均匀稀疏表征的优越性。", prefix="◆ ")
    add_bullet(tf_sum2, "【样本量充盈机制】确立 N=128 的收敛必要性，粉碎高维空间虚假对齐。", prefix="◆ ")

    _, tf_sum3 = add_card(s16, Inches(8.8), Inches(1.15), Inches(3.7), Inches(4.5), "科研边界清晰坦诚", C_GREEN)
    add_bullet(tf_sum3, "【厘清检测边界】诚实报告黑盒蒸馏的阴性结果与中文衍生版的临界状态。", prefix="◆ ")
    add_bullet(tf_sum3, "【学术作风扎实】不掩饰未通过的 2 对极端尾部方差，提供明确可靠的演进解法。", prefix="◆ ")

    c_ack = s16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.85), Inches(11.733), Inches(1.15))
    c_ack.fill.solid()
    c_ack.fill.fore_color.rgb = C_LIGHT_BLUE
    c_ack.line.color.rgb = C_BORDER
    tb_ack = s16.shapes.add_textbox(Inches(1.0), Inches(5.95), Inches(11.333), Inches(0.95))
    tf_a = tb_ack.text_frame
    tf_a.word_wrap = True
    p_a = tf_a.paragraphs[0]
    p_a.alignment = PP_ALIGN.CENTER
    p_a.text = "感谢导师的悉心指导与耐心倾听！"
    p_a.font.name = "Microsoft YaHei"
    p_a.font.size = Pt(16)
    p_a.font.bold = True
    p_a.font.color.rgb = C_BLUE

    p_a2 = tf_a.add_paragraph()
    p_a2.alignment = PP_ALIGN.CENTER
    p_a2.text = "恳请导师批评指正 · 学生 陆顺鑫"
    p_a2.font.name = "Microsoft YaHei"
    p_a2.font.size = Pt(12)
    p_a2.font.color.rgb = C_MUTED
    p_a2.space_before = Pt(4)

    output_path = r"d:\study\Diffip\llm_migration_v4\DiffIP迁移LLM_实验汇报_v4.pptx"
    prs.save(output_path)
    print(f"Presentation built successfully: {output_path}")

    obsidian_dest = r"D:\study\llm-security-study\PPT\实验汇报\DiffIP迁移LLM_实验汇报_v4.pptx"
    try:
        prs.save(obsidian_dest)
        print(f"Also saved copy to: {obsidian_dest}")
    except Exception as e:
        print(f"Failed to copy to obsidian: {e}")

if __name__ == "__main__":
    create_deck()
