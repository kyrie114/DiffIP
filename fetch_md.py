# -*- coding: utf-8 -*-
import sys, io, urllib.request, urllib.parse, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
url = "https://api.github.com/repos/kyrie114/DiffIP/contents/llm_migration_v4"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    items = json.loads(resp.read().decode("utf-8"))
for it in items:
    name = it["name"]
    sz = it["size"]
    if name.endswith(".md"):
        print(f"Downloading {name} ({sz} bytes)")
        dest = os.path.join(r"d:\study\Diffip\llm_migration_v4", name)
        dl_url = "https://raw.githubusercontent.com/kyrie114/DiffIP/main/llm_migration_v4/" + urllib.parse.quote(name)
        urllib.request.urlretrieve(dl_url, dest)
