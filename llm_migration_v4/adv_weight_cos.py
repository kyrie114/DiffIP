"""优点 3/4/5 的最简单参照：直接权重余弦相似度（按参数名逐个对应）。

Usage: python adv_weight_cos.py <name> [name...]     # 名字同 collect.py
受害模型固定为 MAIN["victim"]。对每个嫌疑模型：参数名相同且形状相同才比，
报告这些参数的平均余弦、对上的参数个数 / 受害模型参数个数；一个都对不上记为
"not_comparable"（比如 Sheared 的宽度变了）。删层模型由 build_main 删层后再比，
层号按删后的新编号对应，这正是逐层按编号比权重的做法。
输出 adv_scores/weight_cos.json（追加写）。"""
import json, sys
import torch
from common import H, build_main, load, MAIN

OUT = H / "adv_scores"; OUT.mkdir(exist_ok=True)
F = OUT / "weight_cos.json"


@torch.no_grad()
def cos_table(vsd, ssd):
    vals, matched, emb = [], 0, None
    for k, v in vsd.items():
        s = ssd.get(k)
        if s is None or s.shape != v.shape or v.ndim < 2:
            continue
        a, b = v.flatten().double(), s.flatten().double()
        c = (a @ b / (a.norm() * b.norm())).item()
        vals.append(c); matched += 1
        if "embed_tokens" in k:
            emb = c
    n2d = sum(1 for v in vsd.values() if v.ndim >= 2)
    if not vals:
        return dict(status="not_comparable", matched=0, total=n2d)
    return dict(status="ok", mean_cos=sum(vals) / len(vals), min_cos=min(vals),
                matched=matched, total=n2d, embed_cos=emb)


if __name__ == "__main__":
    res = json.loads(F.read_text()) if F.exists() else {}
    vm, _ = load(MAIN["victim"]); vsd = {k: v.detach().clone() for k, v in vm.state_dict().items()}
    del vm
    for name in sys.argv[1:]:
        if name in res and res[name].get("status") in ("ok", "not_comparable"):
            print("skip", name, flush=True)
            continue
        m, _t, _e, _p = build_main(name)
        res[name] = cos_table(vsd, m.state_dict()); del m
        print(name, res[name], flush=True)
        F.write_text(json.dumps(res, indent=1, ensure_ascii=False))
