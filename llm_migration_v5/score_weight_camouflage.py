"""HuRef and AWM on Llama-2 chat under the three Test-column weight attacks.

Feature attacks for Ours and REEF stay as already measured. This script only
fills HuRef and AWM, then scores Llama-3-8B and MPT-7B as the two negatives
that were not in the Llama-2 static table.

Attacks, matched to the stored feature draw (permutation seed 1, numpy
generator 2026):
  perm: function-preserving residual permutation, RMSNorm permuted with it.
  aniso: fold RMSNorm into the following linears, then diagonal channel scale.
  rot+RMSNorm: fold, orthogonal block rotation, then put a non-constant gamma
  back on the read-side weights so a forward RMSNorm would cancel it. HuRef
  does not see that gamma, which is the T3+T5 mismatch.
"""
from __future__ import annotations

import gc
import json
import os
import time

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import suite_sota as S

CKPT = S.SOTA_DIR / "camouflage_static.json"
EXTRA = S.SOTA_DIR / "static_extra_negatives.json"
READS = ("wq", "wk", "wv", "w1")
WRITES = ("wo", "w2")


def _save(path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _load(path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def attack_matrices(dim: int):
    rng = np.random.default_rng(2026)
    perm_idx = S.make_permutation(dim, seed=1).argmax(0).cpu().numpy().astype(np.int64)
    aniso = np.exp(rng.uniform(-4.0, 4.0, size=(dim,))).astype(np.float32)
    blocks = []
    for _ in range(dim // 64):
        q, _ = np.linalg.qr(rng.standard_normal((64, 64)))
        blocks.append(q.astype(np.float32))
    R = np.zeros((dim, dim), dtype=np.float32)
    for i, q in enumerate(blocks):
        R[i * 64:(i + 1) * 64, i * 64:(i + 1) * 64] = q
    gamma = np.exp(rng.uniform(-2.5, 2.5, size=(dim,))).astype(np.float32)
    return perm_idx, aniso, R, gamma


def _clone_layers(layers: dict) -> dict:
    return {i: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in lw.items()} for i, lw in layers.items()}


def _fold(layers: dict) -> dict:
    out = _clone_layers(layers)
    for lw in out.values():
        g1 = lw["g_attn"]
        g2 = lw["g_mlp"]
        for name in ("wq", "wk", "wv"):
            if name in lw:
                lw[name] = lw[name] * g1[None, :]
        if "w1" in lw:
            lw["w1"] = lw["w1"] * g2[None, :]
        lw["g_attn"] = np.ones_like(g1)
        lw["g_mlp"] = np.ones_like(g2)
    return out


def _mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Eight-thread GEMM. The plain NumPy path stayed on one core."""
    ta = S.torch.from_numpy(np.ascontiguousarray(a))
    tb = S.torch.from_numpy(np.ascontiguousarray(b))
    return (ta.matmul(tb)).numpy()


def _apply(emb, layers, kind, perm_idx, aniso, R, gamma):
    layers = _clone_layers(layers)
    emb = np.ascontiguousarray(emb)
    if kind == "clean":
        return emb, layers
    if kind == "perm":
        emb = emb[:, perm_idx]
        for lw in layers.values():
            for name in READS:
                if name in lw:
                    lw[name] = lw[name][:, perm_idx]
            for name in WRITES:
                if name in lw:
                    lw[name] = lw[name][perm_idx, :]
            lw["g_attn"] = lw["g_attn"][perm_idx]
            lw["g_mlp"] = lw["g_mlp"][perm_idx]
        return emb, layers
    layers = _fold(layers)
    if kind == "aniso":
        emb = emb * aniso
        for lw in layers.values():
            for name in READS:
                if name in lw:
                    lw[name] = lw[name] / aniso[None, :]
            for name in WRITES:
                if name in lw:
                    lw[name] = aniso[:, None] * lw[name]
        return emb, layers
    if kind not in ("rot_base", "rot_rmsnorm"):
        raise ValueError(kind)
    print(f"[apply] {kind} emb {emb.shape}", flush=True)
    emb = _mm(emb, R)
    for i, lw in layers.items():
        for name in READS:
            if name not in lw:
                continue
            print(f"[apply] {kind} layer {i} {name} {lw[name].shape}", flush=True)
            lw[name] = _mm(lw[name], R)
            if kind == "rot_rmsnorm":
                lw[name] = lw[name] / gamma[None, :]
        for name in WRITES:
            if name not in lw:
                continue
            print(f"[apply] {kind} layer {i} {name} {lw[name].shape}", flush=True)
            lw[name] = _mm(R.T, lw[name])
    return emb, layers


def _product_error(emb_a, w_a, emb_b, w_b) -> float:
    xa = emb_a[:64].astype(np.float64)
    xb = emb_b[:64].astype(np.float64)
    pa = xa @ w_a.T.astype(np.float64)
    pb = xb @ w_b.T.astype(np.float64)
    scale = float(np.max(np.abs(pa))) + 1e-8
    return float(np.max(np.abs(pa - pb)) / scale)


def _check_identity(snap, perm_idx, aniso, R, gamma) -> dict:
    # One layer and 64 rows. Checking every stored matrix was a multi-hour GEMM.
    idx = snap["huref_idx"][0]
    tiny = {
        "huref_idx": [idx],
        "emb": np.ascontiguousarray(snap["emb"][:64]),
        "layers": {idx: {"wq": snap["layers"][idx]["wq"], "g_attn": snap["layers"][idx]["g_attn"], "g_mlp": snap["layers"][idx]["g_mlp"]}},
    }
    base_w = tiny["layers"][idx]["wq"]
    folded = _fold(tiny["layers"])
    fold_w = folded[idx]["wq"]
    emb_p, lay_p = _apply(tiny["emb"], tiny["layers"], "perm", perm_idx, aniso, R, gamma)
    emb_a, lay_a = _apply(tiny["emb"], tiny["layers"], "aniso", perm_idx, aniso, R, gamma)
    emb_r, lay_r = _apply(tiny["emb"], tiny["layers"], "rot_base", perm_idx, aniso, R, gamma)
    emb_g, lay_g = _apply(tiny["emb"], tiny["layers"], "rot_rmsnorm", perm_idx, aniso, R, gamma)
    err = {
        "perm": _product_error(tiny["emb"], base_w, emb_p, lay_p[idx]["wq"]),
        "aniso": _product_error(tiny["emb"], fold_w, emb_a, lay_a[idx]["wq"]),
        "rot_before_gamma": _product_error(tiny["emb"], fold_w, emb_r, lay_r[idx]["wq"]),
        "rot_after_gamma": _product_error(tiny["emb"], fold_w, emb_g, lay_g[idx]["wq"]),
    }
    print("[identity] " + " ".join(f"{k}={v:.3e}" for k, v in err.items()), flush=True)
    for key in ("perm", "aniso", "rot_before_gamma"):
        if err[key] > 1e-3:
            raise RuntimeError(f"weight attack {key} does not preserve the folded product ({err[key]:.3e})")
    if err["rot_after_gamma"] < 1e-2:
        raise RuntimeError("non-constant gamma did not change the HuRef product")
    return err


def _extract(key: str):
    model, tok = S.load_suite_model(key)
    layers = S._decoder_stack(model)
    n = len(layers)
    huref_idx = [n - 2, n - 1]
    awm_idx = S.even_layers(n, min(8, n))
    keep = sorted(set(huref_idx + list(awm_idx)))
    stored = {}
    for i in keep:
        layer = layers[i]
        weights = S._extract_layer_weights(model, i)
        # AWM reads Q and K. HuRef also reads V, O, and the MLP, and only on the last two layers.
        names = ("wq", "wk", "wv", "wo", "w1", "w2") if i in huref_idx else ("wq", "wk")
        item = {name: weights[name].detach().float().cpu().numpy() for name in names}
        if hasattr(layer, "input_layernorm"):
            item["g_attn"] = layer.input_layernorm.weight.detach().float().cpu().numpy()
            item["g_mlp"] = layer.post_attention_layernorm.weight.detach().float().cpu().numpy()
        else:
            dim = item["wq"].shape[1]
            item["g_attn"] = np.ones(dim, dtype=np.float32)
            item["g_mlp"] = np.ones(dim, dtype=np.float32)
        stored[i] = item
    emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    snap = {"huref_idx": huref_idx, "awm_idx": list(awm_idx), "emb": emb, "layers": stored}
    del model
    gc.collect()
    print(f"[extract] {key} layers={n} emb={emb.shape}", flush=True)
    return snap, tok


def _scores(values: list[float], seconds: float, extra: dict | None = None) -> dict:
    cell = {
        "scores": [float(v) for v in values],
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "seconds": seconds,
    }
    if extra:
        cell.update(extra)
    return cell


def _huref(vic, sus, tok_v, tok_s) -> tuple[list[float], int]:
    ids_v, ids_s = S._shared_vocab_pairs(tok_v, tok_s, max_tokens=1024, rare_mode=True)
    n_avail = len(ids_v)
    sub_k = min(512, max(64, n_avail // 2))
    ev = vic["emb"]
    es = sus[0]
    sus_layers = sus[1]
    v_layers = [vic["layers"][i] for i in vic["huref_idx"]]
    s_layers = [sus_layers[i] for i in vic["huref_idx"]]

    def invariants(x_hat, lws):
        blocks = []
        for lw in lws:
            q = x_hat @ lw["wq"].T
            k = x_hat @ lw["wk"].T
            ma = q @ k.T
            v = x_hat @ lw["wv"].T
            mb = (v @ lw["wo"].T) @ x_hat.T
            mf = ((x_hat @ lw["w1"].T) @ lw["w2"].T) @ x_hat.T
            blocks.extend([ma.ravel(), mb.ravel(), mf.ravel()])
        return np.concatenate(blocks).astype(np.float64)

    scores = []
    nbytes = 0
    for seed in S.SEED_SETS:
        rng = np.random.default_rng(int(seed))
        sel = rng.choice(n_avail, size=sub_k, replace=False)
        xv = ev[[ids_v[i] for i in sel]].astype(np.float64)
        xs = es[[ids_s[i] for i in sel]].astype(np.float64)
        iv = invariants(xv, v_layers)
        iw = invariants(xs, s_layers)
        nbytes = int(iv.nbytes)
        scores.append(float(np.dot(iv, iw) / (np.linalg.norm(iv) * np.linalg.norm(iw) + 1e-12)))
    return scores, nbytes


def _awm(vic, sus, tok_v, tok_s) -> tuple[list[float], int]:
    ids_v, ids_s = S._shared_vocab_pairs(tok_v, tok_s, max_tokens=4096, rare_mode=False)
    ev = vic["emb"][ids_v].astype(np.float64)
    es = sus[0][ids_s].astype(np.float64)
    evn = ev / (np.linalg.norm(ev, axis=0, keepdims=True) + 1e-12)
    esn = es / (np.linalg.norm(es, axis=0, keepdims=True) + 1e-12)
    cos = evn.T @ esn
    row_ind, col_ind = S.linear_sum_assignment(-np.abs(cos))
    signs = np.sign(cos[row_ind, col_ind])
    signs[signs == 0] = 1.0
    v_idx = vic["awm_idx"]
    s_idx = list(sus[1].keys())
    # Suspect layers keep the victim index set; zip by order of awm_idx.
    s_ids = [i for i in v_idx if i in sus[1]]
    if len(s_ids) != len(v_idx):
        s_ids = list(sus[1].keys())[:len(v_idx)]
    layer_scores = []
    read_bytes = int(ev.nbytes + es.nbytes)
    for lv, ls in zip(v_idx, s_ids):
        wv = vic["layers"][lv]
        ws = sus[1][ls]
        wqv = wv["wq"].T.astype(np.float64)[row_ind, :] * signs[:, None]
        wqs = ws["wq"].T.astype(np.float64)[col_ind, :]
        wkv = wv["wk"].T.astype(np.float64)[row_ind, :] * signs[:, None]
        wks = ws["wk"].T.astype(np.float64)[col_ind, :]
        layer_scores.append(0.5 * (S.unbiased_cka_np(wqv, wqs) + S.unbiased_cka_np(wkv, wks)))
        read_bytes += int(wv["wq"].nbytes + wv["wk"].nbytes + ws["wq"].nbytes + ws["wk"].nbytes)
    scores = []
    for seed in S.SEED_SETS:
        rng = np.random.default_rng(int(seed))
        pick = rng.choice(len(layer_scores), size=max(4, len(layer_scores) - 2), replace=False)
        scores.append(float(np.mean([layer_scores[i] for i in pick])))
    return scores, read_bytes


def _pair(method, vic, emb, layers, tok_v, tok_s):
    t0 = time.time()
    if method == "HuRef":
        scores, nbytes = _huref(vic, (emb, layers), tok_v, tok_s)
    else:
        scores, nbytes = _awm(vic, (emb, layers), tok_v, tok_s)
    return scores, time.time() - t0, nbytes


def _run_attacks(vic, chat, tok_v, tok_s, ckpt: dict) -> dict:
    perm_idx, aniso, R, gamma = attack_matrices(chat["emb"].shape[1])
    if "identity" not in ckpt:
        ckpt["identity"] = _check_identity(chat, perm_idx, aniso, R, gamma)
        _save(CKPT, ckpt)
    for method in ("HuRef", "AWM"):
        done = ckpt.get(method, {})
        if all(k in done for k in ("clean", "perm", "aniso_scale", "rot_rmsnorm")):
            print(f"[skip] {method}", flush=True)
            continue
        block = dict(done)
        for kind, field in (("clean", "clean"), ("perm", "perm"), ("aniso", "aniso_scale"), ("rot_rmsnorm", "rot_rmsnorm")):
            if field in block:
                continue
            print(f"[score] {method} {field}", flush=True)
            emb, layers = _apply(chat["emb"], chat["layers"], kind, perm_idx, aniso, R, gamma)
            scores, sec, nbytes = _pair(method, vic, emb, layers, tok_v, tok_s)
            block[field] = {"scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))}
            print(f"[{method}] {field}={block[field]['mean']:.4f} {sec:.1f}s", flush=True)
            if method == "HuRef":
                block["invariant_bytes"] = nbytes
            else:
                block["tensors_read_bytes"] = nbytes
            if field == "clean":
                block["online_pair_seconds"] = float(sec)
            ckpt[method] = block
            _save(CKPT, ckpt)
            del emb, layers
            gc.collect()
    return ckpt


def _score_negative(vic, tok_v, key: str, extra: dict) -> None:
    if key in extra and "HuRef" in extra[key] and "AWM" in extra[key]:
        print(f"[skip] extra {key}", flush=True)
        return
    snap, tok = _extract(key)
    cell = {}
    for method in ("HuRef", "AWM"):
        scores, sec, _ = _pair(method, vic, snap["emb"], snap["layers"], tok_v, tok)
        cell[method] = _scores(scores, sec)
        print(f"[extra {key}] {method}={cell[method]['mean']:.4f} {sec:.1f}s", flush=True)
        extra[key] = cell
        _save(EXTRA, extra)
    del snap
    gc.collect()


def _merge_camo(ckpt: dict) -> None:
    path = S.SOTA_DIR / "camouflage_test.json"
    camo = _load(path)
    for method in ("HuRef", "AWM"):
        if method in ckpt:
            camo[method] = ckpt[method]
    camo["weight_attack_identity"] = ckpt.get("identity")
    _save(path, camo)


def _synthetic_check() -> None:
    rng = np.random.default_rng(0)
    dim = 128
    emb = rng.standard_normal((80, dim)).astype(np.float32)

    def layer():
        return {
            "wq": rng.standard_normal((dim, dim)).astype(np.float32),
            "wk": rng.standard_normal((dim, dim)).astype(np.float32),
            "wv": rng.standard_normal((dim, dim)).astype(np.float32),
            "wo": rng.standard_normal((dim, dim)).astype(np.float32),
            "w1": rng.standard_normal((dim * 2, dim)).astype(np.float32),
            "w2": rng.standard_normal((dim, dim * 2)).astype(np.float32),
            "g_attn": np.exp(rng.standard_normal(dim) * 0.3).astype(np.float32),
            "g_mlp": np.exp(rng.standard_normal(dim) * 0.3).astype(np.float32),
        }

    snap = {"huref_idx": [0], "awm_idx": [0], "emb": emb, "layers": {0: layer()}}
    perm_idx, aniso, R, gamma = attack_matrices(dim)
    _check_identity(snap, perm_idx, aniso, R, gamma)
    print("[identity] synthetic ok", flush=True)


def main():
    if S.torch is not None and hasattr(S.torch, "set_num_threads"):
        S.torch.set_num_threads(int(os.environ.get("V4_THREADS", "8")))
    _synthetic_check()
    ckpt = _load(CKPT)
    extra = _load(EXTRA)
    need_attacks = any(
        not all(k in ckpt.get(m, {}) for k in ("clean", "perm", "aniso_scale", "rot_rmsnorm"))
        for m in ("HuRef", "AWM")
    )
    need_extra = any(k not in extra or "AWM" not in extra.get(k, {}) for k in ("baseB_victim", "baseD_victim"))
    if not need_attacks and not need_extra:
        _merge_camo(ckpt)
        import fill_measured
        fill_measured.refresh_static_and_report()
        print("CAMO_OK", flush=True)
        return

    vic, tok_v = _extract("baseA_victim")
    if need_attacks:
        chat, tok_s = _extract("baseA_sft_chat")
        ckpt = _run_attacks(vic, chat, tok_v, tok_s, ckpt)
        del chat
        gc.collect()
        _merge_camo(ckpt)
    if need_extra:
        for key in ("baseB_victim", "baseD_victim"):
            _score_negative(vic, tok_v, key, extra)
    del vic
    gc.collect()
    _merge_camo(ckpt)
    import fill_measured
    fill_measured.refresh_static_and_report()
    print("CAMO_OK", flush=True)


if __name__ == "__main__":
    main()
