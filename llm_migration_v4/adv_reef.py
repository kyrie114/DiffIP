"""REEF 同组模型对照（官方设定）：TruthfulQA 前 200 条（random.Random(2024) 打乱后取
前 200，语句 = 问题 + 空格 + 最佳答案，与 0.5B 对照相同），逐条输入、不加聊天模板、
不截断；取每层最后一个 token 的隐藏状态；中心化 + 逐特征标准化后算线性中心化核对齐。
主指标：受害模型第 18 层 vs 嫌疑模型第 18 层；嫌疑模型不足 19 层时取它各层与受害
模型第 18 层的最大值，并记 layer_rule="max"。

Usage:
  python adv_reef.py collect <name> [name...]   # 写 reef_acts/{name}.npz（各层末 token）
  python adv_reef.py score   <name> [name...]   # 写 adv_scores/reef.json
受害模型请先 collect victim。"""
import csv, gc, json, math, random, sys
import numpy as np, torch
from common import H, build_main

ACTS = H / "reef_acts"; ACTS.mkdir(exist_ok=True)
OUT = H / "adv_scores"; OUT.mkdir(exist_ok=True)
TQA_CSV = H.parent / "llm_migration_v3" / "reef_baseline" / "TruthfulQA.csv"
LAYER = 18


def tqa200():
    rows = list(csv.DictReader(open(TQA_CSV, encoding="utf-8")))
    random.Random(2024).shuffle(rows)
    return [r["Question"].strip() + " " + r["Best Answer"].strip() for r in rows[:200]]


@torch.no_grad()
def collect(name):
    f = ACTS / f"{name}.npz"
    if f.exists():
        print("skip", name); return
    m, tok, emb, perm_idx = build_main(name)
    acts = []
    for t in tqa200():
        ids = tok(t, return_tensors="pt").input_ids
        e = emb(ids).detach()
        if perm_idx is not None:
            e = e[..., perm_idx]
        out = m(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], dtype=torch.long),
                use_cache=False, output_hidden_states=True)
        acts.append(torch.stack([h[0, -1] for h in out.hidden_states[1:]]).float().numpy())
    np.savez(f, last=np.stack(acts, 1))  # [layers, 200, D]
    print("saved", f, flush=True); del m; gc.collect()


def std(X):
    X = X - X.mean(0); s = X.std(0, ddof=1); s[s == 0] = 1; return X / s


def lin_cka(X, Y):
    n = X.shape[0]; Hm = np.eye(n) - 1.0 / n
    a, b = Hm @ (X @ X.T) @ Hm, Hm @ (Y @ Y.T) @ Hm
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def score(names):
    F = OUT / "reef.json"
    res = json.loads(F.read_text()) if F.exists() else {}
    V = np.load(ACTS / "victim.npz")["last"].astype(np.float64)
    Xv = std(V[LAYER])
    for name in names:
        S = np.load(ACTS / f"{name}.npz")["last"].astype(np.float64)
        if S.shape[0] > LAYER:
            res[name] = dict(cka=lin_cka(Xv, std(S[LAYER])), layer_rule="18", layers=int(S.shape[0]))
        else:
            c = [lin_cka(Xv, std(S[l])) for l in range(S.shape[0])]
            res[name] = dict(cka=max(c), layer_rule="max", best_layer=int(np.argmax(c)), layers=int(S.shape[0]))
        print(name, res[name], flush=True)
    F.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    torch.set_num_threads(8)
    if sys.argv[1] == "collect":
        for n in sys.argv[2:]:
            collect(n)
    else:
        score(sys.argv[2:])
