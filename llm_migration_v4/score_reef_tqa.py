"""Layer-18 linear CKA on acts_reef_tqa/.

Center, then per-feature standard deviation (ddof=1, zero std -> 1), then
linear CKA. Headline layer is absolute index 18. A model with fewer than 19
layers uses the max CKA of each of its layers against the victim's layer 18.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
A = H / "acts_reef_tqa"
OUT = H / "reef_tqa_result.json"
HEADLINE = 18


def std(X):
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0)
    s = X.std(0, ddof=1)
    s[s == 0] = 1
    return X / s


def lin_cka(X, Y):
    def cent(K):
        n = K.shape[0]
        Hm = np.eye(n) - 1.0 / n
        return Hm @ K @ Hm
    a, b = cent(X @ X.T), cent(Y @ Y.T)
    denom = math.sqrt(float((a * a).sum() * (b * b).sum()))
    if denom == 0:
        raise SystemExit("CKA denominator is 0")
    return float((a * b).sum() / denom)


def load_last(name):
    path = A / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"missing acts_reef_tqa/{name}.npz")
    z = np.load(path)
    last = z["last"]
    total = int(z["total"])
    if last.shape[1] != total or not np.isfinite(last).all():
        raise SystemExit(f"bad acts_reef_tqa/{name}.npz")
    return last


def main():
    names = [p.stem for p in sorted(A.glob("*.npz"))]
    if "victim" not in names:
        raise SystemExit("missing victim activation")
    victim = load_last("victim")
    if victim.shape[1] <= HEADLINE:
        raise SystemExit(f"victim has {victim.shape[1]} layers, need index {HEADLINE}")
    x18 = std(victim[:, HEADLINE, :])
    ident = lin_cka(x18, x18)
    if abs(ident - 1.0) > 1e-6:
        raise SystemExit(f"victim self CKA {ident} is not 1")
    out = {"headline_layer": HEADLINE, "n_statements": int(victim.shape[0]), "models": {}}
    for name in names:
        last = victim if name == "victim" else load_last(name)
        n_layers = int(last.shape[1])
        if n_layers >= HEADLINE + 1:
            score = lin_cka(x18, std(last[:, HEADLINE, :]))
            rule = "absolute_18"
            used = HEADLINE
        else:
            scores = [lin_cka(x18, std(last[:, j, :])) for j in range(n_layers)]
            used = int(np.argmax(scores))
            score = float(scores[used])
            rule = "max_vs_victim_18"
        row = {"linear_cka": score, "layers": n_layers, "hidden": int(last.shape[2]),
               "rule": rule, "suspect_layer": used}
        out["models"][name] = row
        print(f"{name} {score:.4f} layers={n_layers} rule={rule} layer={used}", flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote reef_tqa_result.json", flush=True)


if __name__ == "__main__":
    main()
