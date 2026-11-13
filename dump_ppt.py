import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import zipfile
import re
import xml.etree.ElementTree as ET

def dump_pptx_slides(path):
    print("=== " + path + " ===")
    with zipfile.ZipFile(path, 'r') as z:
        slides = []
        for f in z.namelist():
            m = re.match(r'^ppt/slides/slide(\d+)\.xml$', f)
            if m:
                slides.append((int(m.group(1)), f))
        slides.sort(key=lambda x: x[0])
        print(f"Total slides: {len(slides)}")
        for num, f in slides:
            xml_data = z.read(f)
            root = ET.fromstring(xml_data)
            texts = [elem.text for elem in root.iter() if elem.tag.endswith('}t') and elem.text]
            slide_text = " | ".join(texts)
            print(f"Slide {num}: {slide_text[:160]}")

dump_pptx_slides("d:/study/Diffip/llm_migration_v4/汇报_v4_7B规模验证.pptx")
print("\n" + "="*50 + "\n")
dump_pptx_slides("D:/study/llm-security-study/PPT/实验汇报/DiffIP迁移LLM_实验汇报_v3.pptx")
