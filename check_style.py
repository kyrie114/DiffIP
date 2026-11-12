import zipfile
import xml.etree.ElementTree as ET

with zipfile.ZipFile(r"D:\study\llm-security-study\PPT\实验汇报\DiffIP迁移LLM_实验汇报_v3.pptx", "r") as z:
    for name in ["ppt/slides/slide1.xml", "ppt/slides/slide2.xml"]:
        xml_data = z.read(name)
        root = ET.fromstring(xml_data)
        # find colors (srgbClr)
        colors = set()
        for elem in root.iter():
            if elem.tag.endswith('srgbClr'):
                colors.add(elem.attrib.get('val'))
        print(f"{name} colors:", colors)
