# -*- coding: utf-8 -*-
import sys
import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def build_presentation():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # blank layout

    # Color Palette (Academic Hainan University Style)
    C_BLUE = RGBColor(0, 92, 168)       # #005CA8 主色海大蓝
    C_DARK_BLUE = RGBColor(11, 60, 104) # #0B3C68 深墨蓝
    C_LIGHT_BLUE = RGBColor(235, 243, 250) # #EBF3FA 浅蓝底
    C_TEXT = RGBColor(34, 34, 34)       # #222222 正文字炭黑
    C_MUTED = RGBColor(89, 89, 89)      # #595959 次要文字深灰
    C_BG_CARD = RGBColor(248, 249, 250) # #F8F9FA 卡片浅灰白
    C_BORDER = RGBColor(217, 226, 236)  # #D9E2EC 边框浅蓝灰
    C_WHITE = RGBColor(255, 255, 255)   # 白色
    C_RED = RGBColor(192, 57, 43)       # #C0392B 警示砖红
    C_GREEN = RGBColor(39, 140, 70)     # #278C46 成功墨绿
    C_GOLD = RGBColor(211, 84, 0)       # #D35400 强调琥珀金

    logo_path = r"d:\study\Diffip\llm_migration_v4\hainanu_logo.png"

    def add_header(slide, title, sub_title, slide_num, total_slides=16):
        # Top banner background & line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.85), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = C_BORDER
        line.line.color.rgb = C_BORDER

        # Category / Deck Tracker (Top Left small)
        tx_top = slide.shapes.add_textbox(Inches(0.8), Inches(0.25), Inches(8.0), Inches(0.3))
        tf_top = tx_top.text_frame
        tf_top.word_wrap = True
        tf_top.margin_left = tf_top.margin_right = tf_top.margin_top = tf_top.margin_bottom = 0
        p_top = tf_top.paragraphs[0]
        p_top.text = f"DiffIP迁移LLM · 实验汇报 (v4)  |  {sub_title}"
        p_top.font.name = "Microsoft YaHei"
        p_top.font.size = Pt(10)
        p_top.font.bold = True
        p_top.font.color.rgb = C_BLUE

        # Slide Number (Top Right)
        tx_num = slide.shapes.add_textbox(Inches(11.0), Inches(0.25), Inches(1.0), Inches(0.3))
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

        # Hainan University Logo (Top Right corner)
        if os.path.exists(logo_path):
            slide.shapes.add_picture(logo_path, Inches(12.1), Inches(0.18), Inches(0.46), Inches(0.52))

        # Main Page Title
        tx_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.48), Inches(10.5), Inches(0.4))
        tf_title = tx_title.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.name = "Microsoft YaHei"
        p_title.font.size = Pt(16)
        p_title.font.bold = True
        p_title.font.color.rgb = C_BLUE

        # Bottom subtle footer
        tx_bot = slide.shapes.add_textbox(Inches(0.8), Inches(7.15), Inches(11.733), Inches(0.25))
        tf_bot = tx_bot.text_frame
        tf_bot.margin_left = tf_bot.margin_right = tf_bot.margin_top = tf_bot.margin_bottom = 0
        p_bot = tf_bot.paragraphs[0]
        p_bot.text = "海南大学计算机科学与技术学院 · 大模型安全研究团队 · 2026-11-08"
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
            tb = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.14), width - Inches(0.36), Inches(0.35))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = title
            p.font.name = "Microsoft YaHei"
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = title_color
        return card

    print("Building slides...")

    # ==========================================
    # SLIDE 1: 封面
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    # Background accent bars
    bar1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.12), Inches(3.6))
    bar1.fill.solid()
    bar1.fill.fore_color.rgb = C_BLUE
    bar1.line.fill.background()

    # Top right logo
    if os.path.exists(logo_path):
        s1.shapes.add_picture(logo_path, Inches(11.2), Inches(0.8), Inches(1.2), Inches(1.35))

    # Title box
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

    # Info card
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

    # Save initial test
    prs.save(r"d:\study\Diffip\llm_migration_v4\DiffIP迁移LLM_实验汇报_v4.pptx")
    print("Slide 1 generated.")

build_presentation()
