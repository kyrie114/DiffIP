"""同规则准确率对比：Part 1 同云 CKA 新种子三条规则 + Part 2 REEF 留出。

规则见 试验方案_同规则准确率.md，看到结果前不改。
CKA 公式逐字沿用 exp_gap_cka.py；三条阈值规则逐字沿用 exp_gap_seeds.py；
REEF 层选择与 CKA 口径沿用 llm_migration_v4/score_reef_tqa.py。
不加载模型，只用 feats_newseed/ 与 acts_reef_tqa/ 现有数据。
"""
from __future__ import annotations

import gc
import itertools
import json
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

H = Path(__file__).resolve().parent
V4 = H.parent / "llm_migration_v4"
sys.path.insert(0, str(H.parent))
sys.path.insert(0, str(V4))
from exp_d_prompt_calib import SFT, STRONG, auc  # noqa: E402

N = 128
DIMS = 8
BAND = 0.002
HEADLINE = 18
NEG5 = ["neg_qwen2_5_7b", "neg_llama3_8b", "neg_llama1_7b", "neg_baichuan_7b", "neg_mpt_7b"]
COLLECTED = ["fine_tune", "weight_perm", "prune_dpoff", *STRONG, *NEG5]
CORE_POS = ["fine_tune", *STRONG]
EXTENDED = ["pos_vicuna", "derivative_cn",
            "pos_sheared_1_3b_pruned", "pos_sheared_1_3b", "pos_sheared_1_3b_sharegpt",
            "pos_sheared_2_7b_pruned", "pos_sheared_2_7b", "pos_sheared_2_7b_sharegpt",
            "cpt_codellama", "cpt_wizardmath", "cpt_llemma"]
FEATS = H / "feats_newseed"
ACTS = V4 / "acts_reef_tqa"
OUT_JSON = H / "exp_gap_accuracy_result.json"
OUT_MD = H / "表_同规则准确率对比.md"
ROLE = {"fine_tune": "官方微调", "weight_perm": "权重置换", "rep_perm": "表示置换",
        "scale": "缩放", "prune_dpoff": "删除4层",
        "neg_pruned_sparsegpt": "稀疏剪枝", "neg_pruned_wanda": "稀疏剪枝",
        "neg_pruned_gblm": "稀疏剪枝",
        "neg_qwen2_5_7b": "无关", "neg_llama3_8b": "无关", "neg_llama1_7b": "无关",
        "neg_baichuan_7b": "无关", "neg_mpt_7b": "无关",
        "pos_vicuna": "第三方微调", "derivative_cn": "中文衍生",
        "pos_sheared_1_3b_pruned": "结构化剪枝", "pos_sheared_1_3b": "剪枝后继续训练",
        "pos_sheared_1_3b_sharegpt": "剪枝后指令微调",
        "pos_sheared_2_7b_pruned": "结构化剪枝", "pos_sheared_2_7b": "剪枝后继续训练",
        "pos_sheared_2_7b_sharegpt": "剪枝后指令微调",
        "cpt_codellama": "继续预训练（代码）", "cpt_wizardmath": "继续预训练（数学）",
        "cpt_llemma": "继续预训练（定理）"}


def _std(X):
    X = X - X.mean(0)
    sd = X.std(0, ddof=1)
    sd[sd == 0] = 1
    return X / sd


def lin_cka(X, Y):
    n = X.shape[0]
    Hm = np.eye(n) - 1.0 / n
    a, b = Hm @ (X @ X.T) @ Hm, Hm @ (Y @ Y.T) @ Hm
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def _margins(table, names, anchor, seeds):
    ag = {(r["prompt"], r["seed_set"]): r["similarity"] for r in table[anchor]}
    vals = []
    for name in names:
        g = {(r["prompt"], r["seed_set"]): r["similarity"] for r in table[name]}
        for key, sim in g.items():
            if key[1] in seeds:
                vals.append(sim - ag[key])
    return np.asarray(vals, dtype=np.float64)


def _held(table, anchor, test, t, seeds):
    out = []
    ok = True
    for seed in seeds:
        strong = _margins(table, STRONG, anchor, {seed})
        sft = _margins(table, SFT, anchor, {seed})
        neg = _margins(table, [test], anchor, {seed})
        st = float((strong >= t).mean())
        sft_t = float((sft >= t).mean())
        fpr = float((neg >= t).mean())
        out.append({"strong_tpr": st, "sft_tpr": sft_t, "fpr": fpr})
        if not (st >= 0.90 and sft_t >= 0.90 and fpr <= 0.10):
            ok = False
    return out, ok


def rule_dirs(table, t_of, eval_seeds):
    failed = []
    dirs = []
    for anchor, test in itertools.permutations(NEG5, 2):
        t = float(t_of(table, anchor, test))
        held, ok = _held(table, anchor, test, t, eval_seeds)
        pos = _margins(table, STRONG + SFT, anchor, set(eval_seeds) | {0})
        neg = _margins(table, [test], anchor, set(eval_seeds) | {0})
        dirs.append({"anchor": anchor, "test": test, "t": t, "pass": ok,
                     "held": held, "auc_deploy": auc(pos, neg)})
        if not ok:
            failed.append(f"{anchor}->{test}")
    return {"n_failed": len(failed), "failed": failed, "claim_holds": len(failed) == 0,
            "directions": dirs}


def derive(base, name):
    from llm_migration_v2.transforms import make_anisotropic_scale, make_permutation
    if name == "rep_perm":
        perm = make_permutation(base.shape[-1], seed=1).argmax(0).numpy()
        return base[..., perm]
    scale = np.diag(make_anisotropic_scale(base.shape[-1], seed=0).numpy())
    return base * scale


def part1():
    missing = [n for n in ["victim", *COLLECTED] if not (FEATS / f"main_{n}.npz").exists()]
    if missing:
        raise SystemExit("missing feats_newseed: " + ", ".join(missing))
    arrays = {n: np.load(FEATS / f"main_{n}.npz")["pert"] for n in ["victim", *COLLECTED]}
    arrays["rep_perm"] = derive(arrays["fine_tune"], "rep_perm")
    arrays["scale"] = derive(arrays["fine_tune"], "scale")
    vic = arrays["victim"]
    table = {}
    for name, arr in arrays.items():
        if name == "victim":
            continue
        rows = []
        for p in range(32):
            for k in range(3):
                vals = [lin_cka(_std(vic[p, k, l, :N]), _std(arr[p, k, l, :N]))
                        for l in range(4)]
                rows.append(dict(pair=name, prompt=p, seed_set=k, N=N, dims=DIMS,
                                 similarity=float(np.mean(vals))))
        table[name] = rows
        print(f"cka {name} mean={np.mean([r['similarity'] for r in rows]):.4f}",
              flush=True)
    del arrays, vic
    gc.collect()
    tmax = lambda seeds: (lambda tb, a, t: _margins(tb, [t], a, set(seeds)).max())
    res = {
        "seeds": "3123/4123/5123 relabeled 0/1/2",
        "band": BAND,
        "max1": rule_dirs(table, tmax({0}), (1, 2)),
        "band_rule": rule_dirs(table, lambda tb, a, t: tmax({0})(tb, a, t) + BAND, (1, 2)),
        "max2": rule_dirs(table, tmax({0, 1}), (2,)),
    }
    res["band_confirmed"] = bool(res["band_rule"]["claim_holds"])
    return res


def part2():
    names = [p.stem for p in sorted(ACTS.glob("*.npz"))]
    if "victim" not in names:
        raise SystemExit("missing acts_reef_tqa/victim.npz")
    last = {n: np.load(ACTS / f"{n}.npz")["last"] for n in names}
    need = ["victim", *CORE_POS, *NEG5, *EXTENDED]
    miss = [n for n in need if n not in names]
    if miss:
        raise SystemExit("missing acts_reef_tqa: " + ", ".join(miss))
    x18_all = _std(last["victim"][:, HEADLINE, :])
    even = np.arange(0, 200, 2)
    odd = np.arange(1, 200, 2)
    models = {}
    for n in names:
        X = last[n]
        L = int(X.shape[1])
        if n == "victim":
            models[n] = {"rule": "absolute_18", "layer": HEADLINE,
                         "cka_even": 1.0, "cka_odd": 1.0}
            continue
        if L > HEADLINE:
            layer, rule = HEADLINE, "absolute_18"
        else:
            sc = [lin_cka(x18_all, _std(X[:, j, :])) for j in range(L)]
            layer, rule = int(np.argmax(sc)), "max_vs_victim_18"
        ce = lin_cka(_std(last["victim"][even, HEADLINE, :]), _std(X[even, layer, :]))
        co = lin_cka(_std(last["victim"][odd, HEADLINE, :]), _std(X[odd, layer, :]))
        models[n] = {"rule": rule, "layer": layer, "cka_even": ce, "cka_odd": co}
        print(f"reef {n} even={ce:.4f} odd={co:.4f} {rule} L{layer}", flush=True)
    del last
    gc.collect()
    tE = max(models[n]["cka_even"] for n in NEG5)
    tO = max(models[n]["cka_odd"] for n in NEG5)

    def direction(t, key):
        pos = {n: models[n][key] for n in CORE_POS}
        neg = {n: models[n][key] for n in NEG5}
        hit_p = {n: bool(v >= t) for n, v in pos.items()}
        hit_n = {n: bool(v >= t) for n, v in neg.items()}
        n_pos = sum(hit_p.values())
        n_neg = sum(hit_n.values())
        return {"t": float(t),
                "pos_recall": f"{n_pos}/{len(pos)}",
                "fpr": f"{n_neg}/{len(neg)}",
                "pass": bool(n_pos == len(pos) and n_neg == 0),
                "by_pos": {n: {"cka": v, "hit": hit_p[n]} for n, v in pos.items()},
                "by_neg": {n: {"cka": v, "hit": hit_n[n]} for n, v in neg.items()}}

    per_model = {}
    for n in names:
        if n == "victim":
            continue
        per_model[n] = {
            "role": ROLE.get(n, ""),
            "rule": models[n]["rule"], "layer": models[n]["layer"],
            "cka_even": models[n]["cka_even"], "cka_odd": models[n]["cka_odd"],
            "abs_diff": abs(models[n]["cka_even"] - models[n]["cka_odd"]),
            "margin_odd_vs_tE": models[n]["cka_odd"] - tE,
            "margin_even_vs_tO": models[n]["cka_even"] - tO,
        }
    return {
        "split": "statement index even(0,2,..198)/odd(1,3,..199), 100 each",
        "headline": {"rule": "absolute 18 if layers>18 else max vs victim 18"},
        "t_even_max_neg5": tE, "t_odd_max_neg5": tO,
        "E2O": direction(tE, "cka_odd"), "O2E": direction(tO, "cka_even"),
        "models": per_model,
    }


def refs():
    diffip = json.loads((H / "exp_gap_seeds_result.json").read_text())
    def brief(k):
        r = diffip[k]
        return {"n_failed": r["n_failed"], "failed": r["failed"],
                "claim_holds": r["claim_holds"]}
    reef_full = json.loads((V4 / "reef_tqa_result.json").read_text())
    gap32 = json.loads((H / "exp_gap_reef_result.json").read_text())
    return {
        "diffip_newseed": {"max1": brief("max1"), "band_rule": brief("band_rule"),
                           "max2": brief("max2")},
        "reef_full200": {n: r["linear_cka"] for n, r in reef_full["models"].items()},
        "reef_gap32": {"gap32": gap32["gap32"], "pos_min": gap32["pos_min"],
                       "neg_max": gap32["neg_max"], "sign_stable": gap32["sign_stable"]},
    }


def brief_str(r):
    return f"{20 - r['n_failed']}/20" + ("" if not r["failed"]
                                          else "（" + ", ".join(r["failed"]) + "）")


def write_md(p1, p2, rf):
    L = []
    L.append("# 同规则准确率对比（Llama-2-7B 面板）\n")
    L.append("Part 1/2 本轮新算（数据：feats_newseed/ 与 acts_reef_tqa/，纯 numpy，不加载模型）；"
             "Part 3 沿用已存结果。通过线在看到结果前写死，见 `试验方案_同规则准确率.md`。\n")
    L.append("## Part 1 同云线性 CKA 新种子确认（NEG5 两两有序 20 对，种子 3123/4123/5123）\n")
    L.append("| 规则 | CKA（本轮） | DiffIP（沿用 exp_gap_seeds_result.json） |")
    L.append("|---|---|---|")
    for key, label in (("max1", "max1：种子0最大间隔，检验种子1、2"),
                       ("band_rule", "band：max1 再加 0.002，检验种子1、2"),
                       ("max2", "max2：种子0、1合取最大，只检验种子2")):
        L.append(f"| {label} | {brief_str(p1[key])} | {brief_str(rf['diffip_newseed'][key])} |")
    L.append("")
    L.append("判定口径与 Part 1 的 DiffIP 完全相同：每个检验种子上 strong/SFT 召回 ≥0.90 且 "
             "deploy 误报 ≤0.10，20 对全过才主张成立。\n")
    L.append("## Part 2 REEF 留出准确率（TQA-200 按偶/奇下标各 100 条，头条层 18）\n")
    L.append("| 方向 | 阈值 t（校准半上 5 无关最大 CKA） | 核心正例召回 | 无关误报 | 通过（4/4 且 0/5） |")
    L.append("|---|---|---|---|---|")
    for key, label in (("E2O", "偶→奇（偶数标定）"), ("O2E", "奇→偶（奇数标定）")):
        d = p2[key]
        L.append(f"| {label} | {d['t']:.4f} | {d['pos_recall']} | {d['fpr']} | "
                 f"{'通过' if d['pass'] else '不通过'} |")
    L.append("")
    L.append("核心正例 = 官方微调 + 三种同底座稀疏剪枝。扩展类只报告，不计入通过判定。\n")
    L.append("### 逐模型 CKA（偶/奇半）与对两个阈值的间隔\n")
    L.append("| 模型 | 类别 | 层 | CKA_偶 | CKA_奇 | \\|差\\| | 奇 − t_偶 | 偶 − t_奇 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for n in ["fine_tune", *STRONG, "pos_vicuna", "derivative_cn", *NEG5,
              "pos_sheared_1_3b_pruned", "pos_sheared_1_3b", "pos_sheared_1_3b_sharegpt",
              "pos_sheared_2_7b_pruned", "pos_sheared_2_7b", "pos_sheared_2_7b_sharegpt",
              "cpt_codellama", "cpt_wizardmath", "cpt_llemma"]:
        m = p2["models"][n]
        L.append(f"| {n} | {m['role']} | {m['layer']} | {m['cka_even']:.4f} | "
                 f"{m['cka_odd']:.4f} | {m['abs_diff']:.4f} | {m['margin_odd_vs_tE']:+.4f} | "
                 f"{m['margin_even_vs_tO']:+.4f} |")
    L.append("")
    L.append("## Part 3 参照（沿用，不重算）\n")
    L.append("- DiffIP 旧种子 20 对（exp_gap_cka 同口径）：19/20，失败 "
             "neg_qwen2_5_7b->neg_llama3_8b（见 exp_gap_cka_result.json）。")
    L.append("- REEF 全 200 单数：见 `reef_tqa_result.json`（本表 Part 2 是其留出化）。")
    L.append("- REEF 32 探针 gap：{:.4f}（pos_min {:.4f} − neg_max {:.4f}，符号稳定 {}，"
             "见 exp_gap_reef_result.json）。\n".format(
                 rf["reef_gap32"]["gap32"], rf["reef_gap32"]["pos_min"],
                 rf["reef_gap32"]["neg_max"], rf["reef_gap32"]["sign_stable"]))
    L.append("## 结果（机械汇总，解读见会话汇报）\n")
    L.append(f"- CKA 新种子：max1 {brief_str(p1['max1'])}，band {brief_str(p1['band_rule'])}，"
             f"max2 {brief_str(p1['max2'])}。")
    L.append(f"- REEF 留出：E→O 召回 {p2['E2O']['pos_recall']} / 误报 {p2['E2O']['fpr']}"
             f"（{'通过' if p2['E2O']['pass'] else '不通过'}）；O→E 召回 {p2['O2E']['pos_recall']}"
             f" / 误报 {p2['O2E']['fpr']}（{'通过' if p2['O2E']['pass'] else '不通过'}）。")
    OUT_MD.write_text("\n".join(L))
    print(f"wrote {OUT_MD.name}", flush=True)


def main():
    p1 = part1()
    p2 = part2()
    rf = refs()
    result = {"part1_cka_newseed": p1, "part2_reef_heldout": p2, "part3_reference": rf}
    OUT_JSON.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {OUT_JSON.name}", flush=True)
    write_md(p1, p2, rf)
    print("DONE",
          "cka max1/band/max2 =",
          p1["max1"]["claim_holds"], p1["band_confirmed"], p1["max2"]["claim_holds"],
          "reef E2O/O2E =", p2["E2O"]["pass"], p2["O2E"]["pass"], flush=True)


if __name__ == "__main__":
    main()
