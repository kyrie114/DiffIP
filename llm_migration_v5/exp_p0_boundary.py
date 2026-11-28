"""P0 boundary observations. Calls the existing HuRef and AWM functions. No new forward."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import suite_sota as S  # noqa: E402
import exp_p0_expanded_pool as P  # noqa: E402

FINISHED = [
    "pos_fusellm_7b",
    "pos_abel_7b_002",
    "neg_openllama_7b_v2",
    "neg_olmo2_7b",
    "neg_bloom_7b1",
    "neg_pythia_6_9b",
    "neg_opt_6_7b",
    "neg_redpajama_7b",
    "neg_chatglm2_6b",
    "neg_openchat_3_5",
    "neg_starling_7b",
]
POSITIVES = ("pos_fusellm_7b", "pos_abel_7b_002")
NEGATIVES = [k for k in FINISHED if k not in POSITIVES]
OUT = H / "sota_scores" / "p0_boundary_result.json"


def _pin_threads() -> None:
    if S.torch is not None:
        S.torch.set_num_threads(1)


def _load_4(key: str) -> np.ndarray:
    if key in P.P0_MODELS:
        path = P.locate_p0_ours(key)
        if path is None:
            raise FileNotFoundError(key)
        arr = np.load(path)["pert"]
    else:
        arr = S.load_ours_4layers(key)
    if arr.ndim != 5 or arr.shape[2] != 4:
        raise RuntimeError(f"{key} feature has no 4-layer axis: {arr.shape}")
    return arr


def _layer_means(sus: np.ndarray, vic: np.ndarray) -> tuple[list[float], list[float]]:
    raw, norm = S.per_layer_joint_scores(sus, vic, dims=8)
    if not np.isfinite(raw).all() or not np.isfinite(norm).all():
        raise RuntimeError("non-finite per-layer score")
    return raw.mean(axis=(0, 1)).tolist(), norm.mean(axis=(0, 1)).tolist()


def layer_report() -> dict:
    _pin_threads()
    # project_sequences uses sklearn's randomized SVD. One seeded stream keeps two launches equal.
    np.random.seed(0)
    vic = _load_4("baseA_victim")
    per = {}
    band_raw = []
    band_norm = []
    for key in FINISHED:
        raw, norm = _layer_means(_load_4(key), vic)
        per[key] = {"raw_mean": raw, "norm_mean": norm}
        print(
            f"[layer] {key} raw=" + " ".join(f"{v:.4f}" for v in raw)
            + " norm=" + " ".join(f"{v:.4f}" for v in norm),
            flush=True,
        )
        if key in NEGATIVES:
            band_raw.append(raw)
            band_norm.append(norm)
    raw_m = np.asarray(band_raw)
    norm_m = np.asarray(band_norm)
    return {
        "victim": "baseA_victim",
        "positives": {
            key: per[key] for key in POSITIVES
        },
        "pos_fusellm_7b": per["pos_fusellm_7b"],
        "pos_abel_7b_002": per["pos_abel_7b_002"],
        "negatives": {key: per[key] for key in NEGATIVES},
        "unrelated_band": {
            "raw_min": raw_m.min(axis=0).tolist(),
            "raw_max": raw_m.max(axis=0).tolist(),
            "norm_min": norm_m.min(axis=0).tolist(),
            "norm_max": norm_m.max(axis=0).tolist(),
        },
    }


def _blank() -> dict:
    return {
        "thresholds_fit_on_these_models": False,
        "models": list(FINISHED),
        "per_layer": {},
        "weights": {},
    }


def _read() -> dict:
    if not OUT.exists():
        return _blank()
    data = json.loads(OUT.read_text(encoding="utf-8"))
    data.setdefault("weights", {})
    data.setdefault("per_layer", {})
    data["thresholds_fit_on_these_models"] = False
    data["models"] = list(FINISHED)
    return data


def _write(data: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, OUT)


def cmd_layers() -> None:
    data = _read()
    data["per_layer"] = layer_report()
    _write(data)
    print(f"[layer] saved {OUT.name}", flush=True)


def _score_loaded(m_vic, tok_vic, m_sus, tok_sus) -> dict:
    h_scores, h_sec = S.compute_huref_ics(m_vic, tok_vic, m_sus, tok_sus)
    a_scores, a_sec = S.compute_awm_ucka(m_vic, tok_vic, m_sus, tok_sus)
    return {
        "HuRef": {
            "scores": h_scores,
            "mean": float(np.mean(h_scores)),
            "std": float(np.std(h_scores)),
            "seconds": h_sec,
        },
        "AWM": {
            "scores": a_scores,
            "mean": float(np.mean(a_scores)),
            "std": float(np.std(a_scores)),
            "seconds": a_sec,
        },
    }


def cmd_weights() -> None:
    _pin_threads()
    data = _read()
    if "pos_fusellm_7b" not in data["per_layer"]:
        data["per_layer"] = layer_report()
        _write(data)
    pending = [k for k in FINISHED if k not in data["weights"]]
    if not pending:
        for key in FINISHED:
            row = data["weights"][key]
            print(f"[weight] {key} HuRef={row['HuRef']['mean']:.4f} AWM={row['AWM']['mean']:.4f}", flush=True)
        print("[weight] all 11 already stored", flush=True)
        return
    m_vic, tok_vic = S.load_suite_model("baseA_victim")
    for key in pending:
        m_sus, tok_sus = P._load_p0_model(P.P0_MODELS[key]["path"])
        row = _score_loaded(m_vic, tok_vic, m_sus, tok_sus)
        if not (math.isfinite(row["HuRef"]["mean"]) and math.isfinite(row["AWM"]["mean"])):
            raise RuntimeError(f"non-finite weight score for {key}")
        data["weights"][key] = row
        _write(data)
        print(
            f"[weight] {key} HuRef={row['HuRef']['mean']:.4f} ({row['HuRef']['seconds']:.1f}s) "
            f"AWM={row['AWM']['mean']:.4f} ({row['AWM']['seconds']:.1f}s)",
            flush=True,
        )
        del m_sus
        import gc
        gc.collect()
    del m_vic


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "layers":
        cmd_layers()
    elif cmd == "weights":
        cmd_weights()
    else:
        raise SystemExit("usage: exp_p0_boundary.py layers|weights")


if __name__ == "__main__":
    main()
