import os
import win32com.client

ppt_path = os.path.abspath(r"d:\study\Diffip\llm_migration_v4\DiffIP迁移LLM_实验汇报_v4.pptx")
out_dir = os.path.abspath(r"d:\study\Diffip\llm_migration_v4\_verify")
os.makedirs(out_dir, exist_ok=True)

try:
    powerpoint = win32com.client.DispatchEx("PowerPoint.Application")
    # ppFixedFormatTypePNG is not directly available, but presentation.Export(out_dir, "PNG", 1920, 1080) exports all slides
    deck = powerpoint.Presentations.Open(ppt_path, WithWindow=False)
    deck.Export(out_dir, "PNG", 1920, 1080)
    deck.Close()
    powerpoint.Quit()
    print("Exported slides via COM successfully!")
    for f in os.listdir(out_dir):
        if f.endswith(".PNG") or f.endswith(".png"):
            print("Exported:", f)
except Exception as e:
    print(f"COM Export error: {e}")
