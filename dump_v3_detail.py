import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import zipfile
import re
import xml.etree.ElementTree as ET

with zipfile.ZipFile(r"D:\study\llm-security-study\PPT\实验汇报\DiffIP迁移LLM_实验汇报_v3.pptx", "r") as z:
    slides = []
    for f in z.namelist():
        m = re.match(r"^ppt/slides/slide(\d+)\.xml$", f)
        if m:
            slides.append((int(m.group(1)), f))
    slides.sort(key=lambda x: x[0])
    for num, f in slides:
        xml_data = z.read(f)
        root = ET.fromstring(xml_data)
        texts = [elem.text for elem in root.iter() if elem.tag.endswith("}t") and elem.text]
        print(f"\n--- Slide {num} ---")
        print("\n".join(texts[:10]))
