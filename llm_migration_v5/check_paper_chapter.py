"""Check the experimental chapter against stored JSON. Exit 0 only when every cited headline matches."""
from __future__ import annotations

import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parent
CHAPTER = H / "docs" / "实验章节.md"
FOUR = H / "sota_scores" / "summary_metrics.json"
P0 = H / "sota_scores" / "p0_expanded_pool_result.json"
UNREL = ROOT / "llm_migration_v4" / "exp_unrel_result.json"
SEEDS = H / "exp_gap_seeds_result.json"
CKA = H / "exp_gap_cka_result.json"
QWEN3 = H / "exp_gap_qwen3_result.json"
BOUNDARY = H / "sota_scores" / "p0_boundary_result.json"

FORBIDDEN = (
    "0.6718",
    "0.838/0.844/0.860/0.712",
    "16 KB",
    "16KB",
    "1.48",
    "严格不变",
    "drop-6",
    "drop6",
)


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    raise SystemExit(1)


def load(path: Path):
    if not path.exists():
        fail(f"missing {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(text: str, snippet: str) -> None:
    if snippet not in text:
        fail(f"chapter missing {snippet}")


def pooled_accuracy(unrel: dict) -> str:
    tp = fn = fp = tn = 0.0
    for d in unrel["directions"]:
        for s in range(2):
            tp += d["held_strong_tpr"][s] * 96 + d["held_sft_tpr"][s] * 160
            fn += (1 - d["held_strong_tpr"][s]) * 96 + (1 - d["held_sft_tpr"][s]) * 160
            fp += d["held_fpr"][s] * 32
            tn += (1 - d["held_fpr"][s]) * 32
    correct = tp + tn
    total = tp + fn + fp + tn
    return f"{int(round(correct))}/{int(round(total))}"


def main() -> None:
    if not CHAPTER.exists():
        print("FAIL chapter absent")
        raise SystemExit(1)
    text = CHAPTER.read_text(encoding="utf-8")
    for bad in FORBIDDEN:
        if bad in text:
            fail(f"forbidden measurement text {bad}")
    if "20/20" in text and "冻结" in text and "DiffIP 冻结门限通过 20/20" in text:
        fail("frozen gate written as 20/20")

    unrel = load(UNREL)
    passed = unrel["n_directions"] - unrel["n_failed"]
    require(text, f"{passed}/{unrel['n_directions']}")
    acc = pooled_accuracy(unrel)
    require(text, acc)
    print(f"20-pair {passed}/{unrel['n_directions']} accuracy {acc}")

    seeds = load(SEEDS)
    for rule, label in (("max1", "max1"), ("band_rule", "band")):
        n = sum(1 for d in seeds[rule]["directions"] if d["pass"])
        require(text, f"{label} {n}/20")
        if rule == "max1" and n == 20:
            fail("max1 unexpectedly passed")
    if seeds.get("band_confirmed"):
        fail("seed band_confirmed is true in source")
    require(text, "band 未确认")

    cka = load(CKA)
    cka_pass = cka["cka"]["n_directions"] - cka["cka"]["n_failed"]
    cos_pass = cka["cosine_same_d"]["n_directions"] - cka["cosine_same_d"]["n_failed"]
    require(text, f"同云线性 CKA {cka_pass}/20")
    require(text, f"同维扰动余弦 {cos_pass}/12")
    qwen = load(QWEN3)
    q_pass = len(qwen["directions"]) - qwen["n_failed"]
    require(text, f"Qwen3 第二底座 {q_pass}/12")
    if cka["cka"]["claim_holds"] or cka["cosine_same_d"]["claim_holds"] or qwen["claim_holds"]:
        fail("a failed pre-registered file is marked claim_holds")
    if "同云线性 CKA 20/20，通过" in text:
        fail("same-cloud CKA written as passed")

    four = load(FOUR)
    role_order = ("pruning", "quantization", "finetune", "domain")
    means = []
    for role in role_order:
        means.append(f"{four['by_role_aggregated'][role]['Ours']['auc_mean']:.4f}")
    require(text, " / ".join(means))
    print("four-base Ours " + " / ".join(means))
    cam = four["camouflage_test"]["Ours"]
    require(text, f"{cam['perm']['mean']:.4f}")
    require(text, f"{cam['aniso_scale']['mean']:.4f}")
    require(text, f"{cam['rot_rmsnorm']['mean']:.4f}")

    p0 = load(P0)
    fuse = p0["ours"]["pos_fusellm_7b"]
    abel = p0["ours"]["pos_abel_7b_002"]
    fuse_auc = f"{fuse['auc_raw_vs_expanded']:.4f} / {fuse['auc_norm_vs_expanded']:.4f}"
    abel_auc = f"{abel['auc_raw_vs_expanded']:.4f} / {abel['auc_norm_vs_expanded']:.4f}"
    require(text, fuse_auc)
    require(text, abel_auc)
    print(f"FuseLLM AUC {fuse_auc}")
    print(f"Abel AUC {abel_auc}")
    require(text, f"{p0['summary']['new_neg_frozen20_mean_fpr']:.4f}")
    oc = p0["reef"]["neg_openchat_3_5"]["cka_full200"]
    st = p0["reef"]["neg_starling_7b"]["cka_full200"]
    if oc != st:
        fail("OpenChat and Starling REEF full CKA are not bitwise equal in JSON")
    require(text, f"{oc:.6f}")
    require(text, "逐位相同")
    require(text, "internlm2-base-7b")
    require(text, "非有限")
    if "internlm2-base-7b" in text and "AUC" in text.split("internlm2-base-7b")[1][:80]:
        fail("internlm2 has an imputed score nearby")

    if not BOUNDARY.exists():
        fail("boundary result absent")
    boundary = load(BOUNDARY)
    if boundary.get("thresholds_fit_on_these_models"):
        fail("thresholds were fit on the 11 models")
    layers = boundary["per_layer"]
    for key in ("pos_fusellm_7b", "pos_abel_7b_002"):
        raw = layers[key]["raw_mean"]
        norm = layers[key]["norm_mean"]
        if len(raw) != 4 or len(norm) != 4:
            fail(f"{key} does not have four layer scores")
        if not all(map(_finite, raw + norm)):
            fail(f"{key} layer score is not finite")
        require(text, " ".join(f"{v:.4f}" for v in raw))
        require(text, " ".join(f"{v:.4f}" for v in norm))
    print("per-layer FuseLLM " + " ".join(f"{v:.4f}" for v in layers["pos_fusellm_7b"]["raw_mean"]))
    print("per-layer Abel " + " ".join(f"{v:.4f}" for v in layers["pos_abel_7b_002"]["raw_mean"]))
    weights = boundary["weights"]
    expected = [
        "pos_fusellm_7b", "pos_abel_7b_002", "neg_openllama_7b_v2", "neg_olmo2_7b",
        "neg_bloom_7b1", "neg_pythia_6_9b", "neg_opt_6_7b", "neg_redpajama_7b",
        "neg_chatglm2_6b", "neg_openchat_3_5", "neg_starling_7b",
    ]
    if "neg_internlm2_base_7b" in weights or "internlm2" in json.dumps(weights):
        fail("internlm2 present in boundary weights")
    if list(weights) != expected and set(weights) != set(expected):
        fail("weight keys are not the 11 finished models")
    for key in expected:
        h = weights[key]["HuRef"]["mean"]
        a = weights[key]["AWM"]["mean"]
        if not (_finite(h) and _finite(a)):
            fail(f"{key} weight score is not finite")
        require(text, f"{h:.4f}")
        require(text, f"{a:.4f}")
        print(f"weight {key} HuRef {h:.4f} AWM {a:.4f}")
    print("CHECK_OK")


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and v == v and abs(v) != float("inf")


if __name__ == "__main__":
    main()
