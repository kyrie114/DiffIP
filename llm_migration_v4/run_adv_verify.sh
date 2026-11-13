#!/usr/bin/env bash
# 2026-11-12 advantage verification. Sequential, CPU only, resumable.
# Decision rule stays inside adv_fixed_threshold.py. This script only
# collects the three layer-drop chats and runs the four pulled scripts.
set -euo pipefail
cd "$(dirname "$0")"
trap 'echo "FAILED: line $LINENO status $?"' ERR

PY="${PY:-python3}"
unset V4_FEAT_DIR V4_SEED_SETS V4_ALL_LAYERS V4_LAYER_SUBSET
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export V4_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export V4_SCORES_DIR=scores_panel
export CUDA_VISIBLE_DEVICES=""

NEG=(neg_qwen2_5_7b neg_llama3_8b neg_llama1_7b neg_baichuan_7b neg_mpt_7b)
GATED=(fine_tune neg_pruned_sparsegpt neg_pruned_wanda neg_pruned_gblm
       prune_dpoff weight_perm rep_perm scale)
ADV=(pos_vicuna
     pos_sheared_1_3b_pruned pos_sheared_1_3b pos_sheared_1_3b_sharegpt
     pos_sheared_2_7b_pruned pos_sheared_2_7b pos_sheared_2_7b_sharegpt
     cpt_codellama cpt_wizardmath cpt_llemma
     drop2_chat drop8_chat drop12_chat)
COS=(victim fine_tune derivative_cn
     neg_pruned_sparsegpt neg_pruned_wanda neg_pruned_gblm
     prune_dpoff weight_perm
     "${ADV[@]}"
     "${NEG[@]}")
REEF_NAMES=(victim fine_tune derivative_cn
            neg_pruned_sparsegpt neg_pruned_wanda neg_pruned_gblm
            prune_dpoff weight_perm
            "${ADV[@]}"
            "${NEG[@]}")

mkdir -p logs adv_scores feats scores scores_unrel scores_panel acts_reef_tqa reef_acts figs_adv
# scores/summary_main.json is a symlink into the existing roster. Record its
# stamp so a write through that link is visible at the end.
if [[ -L scores/summary_main.json || -f scores/summary_main.json ]]; then
  SUM_STAMP=$(stat -L -c %Y scores/summary_main.json)
else
  SUM_STAMP=""
fi
echo "ADV_BEGIN $(date -Is)"

echo "--- gated fixed threshold (existing scores) ---"
"$PY" adv_fixed_threshold.py --neg "${NEG[@]}" --pos "${GATED[@]}"
cp -f adv_scores/fixed_threshold.json adv_scores/fixed_threshold_gated.json

echo "--- transpose existing TruthfulQA activations into reef_acts/ ---"
"$PY" - <<'PY'
import numpy as np
from pathlib import Path
src, dst = Path("acts_reef_tqa"), Path("reef_acts")
dst.mkdir(exist_ok=True)
n = 0
for p in sorted(src.glob("*.npz")):
    out = dst / p.name
    if out.exists():
        z = np.load(out)
        last = z["last"]
        if last.ndim != 3 or last.shape[1] != 200 or not np.isfinite(last).all():
            raise SystemExit(f"bad reef_acts/{p.name}")
        print("skip", p.name, tuple(last.shape), flush=True)
        n += 1
        continue
    last = np.load(p)["last"]  # [200, layers, D]
    if last.ndim != 3 or last.shape[0] != 200 or not np.isfinite(last).all():
        raise SystemExit(f"bad acts_reef_tqa/{p.name}")
    arr = np.transpose(last, (1, 0, 2))
    np.savez(out, last=arr)
    print("converted", p.name, tuple(arr.shape), flush=True)
    n += 1
print("reef_converted", n, flush=True)
PY

echo "--- collect layer-drop chats ---"
for name in drop2_chat drop8_chat drop12_chat; do
  if [[ -f feats/main_${name}.npz ]]; then
    echo "skip collect $name"
    continue
  fi
  "$PY" collect.py main "$name"
done

echo "--- score layer-drop chats into scores_panel/ ---"
need_score=()
for name in drop2_chat drop8_chat drop12_chat; do
  if [[ ! -f scores_panel/main_${name}.json ]]; then
    need_score+=("$name")
  fi
done
if ((${#need_score[@]})); then
  "$PY" score.py main "${need_score[@]}"
else
  echo "skip score; three drop rows exist"
fi

echo "--- weight cosine ---"
"$PY" adv_weight_cos.py "${COS[@]}"

echo "--- REEF collect, one process each, for models without a cache ---"
for name in prune_dpoff weight_perm drop2_chat drop8_chat drop12_chat; do
  if [[ -f reef_acts/${name}.npz ]]; then
    echo "skip reef collect $name"
    continue
  fi
  "$PY" adv_reef.py collect "$name"
done
echo "--- REEF score ---"
"$PY" adv_reef.py score "${REEF_NAMES[@]}"

echo "--- adv fixed threshold ---"
"$PY" adv_fixed_threshold.py --neg "${NEG[@]}" --pos "${ADV[@]}"
cp -f adv_scores/fixed_threshold.json adv_scores/fixed_threshold_adv.json

echo "--- layer residuals ---"
residual_state=$("$PY" - <<'PY'
import json
from pathlib import Path
need = """victim fine_tune derivative_cn neg_pruned_sparsegpt neg_pruned_wanda neg_pruned_gblm prune_dpoff weight_perm pos_vicuna pos_sheared_1_3b_pruned pos_sheared_1_3b pos_sheared_1_3b_sharegpt pos_sheared_2_7b_pruned pos_sheared_2_7b pos_sheared_2_7b_sharegpt cpt_codellama cpt_wizardmath cpt_llemma drop2_chat drop8_chat drop12_chat neg_qwen2_5_7b neg_llama3_8b neg_llama1_7b neg_baichuan_7b neg_mpt_7b""".split()
p = Path("adv_scores/layer_residual.json")
if not p.is_file():
    print("RESIDUAL_MISSING")
    raise SystemExit(0)
have = set(json.loads(p.read_text()))
missing = [n for n in need if n not in have]
print("RESIDUAL_MISSING" if missing else "RESIDUAL_COMPLETE")
PY
)
if [[ "$residual_state" == "RESIDUAL_COMPLETE" ]]; then
  echo "skip layer residual"
else
  V4_THREADS=2 "$PY" adv_layer_residual.py \
    victim fine_tune derivative_cn \
    neg_pruned_sparsegpt neg_pruned_wanda neg_pruned_gblm \
    prune_dpoff weight_perm \
    "${ADV[@]}" \
    "${NEG[@]}"
fi

echo "--- report ---"
"$PY" exp_adv_report.py
if [[ -n "${SUM_STAMP}" ]]; then
  now=$(stat -L -c %Y scores/summary_main.json)
  if [[ "$now" != "$SUM_STAMP" ]]; then
    echo "FAILED: scores/summary_main.json changed during the run"
    exit 1
  fi
fi
echo "ADV_DONE $(date -Is)"
