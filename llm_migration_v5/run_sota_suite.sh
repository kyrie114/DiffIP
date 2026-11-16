#!/usr/bin/env bash
# V5 paper suite: 4 bases x 4 derivation types, then static baselines,
# mechanism, evaluate, and plots. Decision rules stay inside suite_sota.py.
# CPU only. One model per process. Existing npz files are skipped.
set -euo pipefail
cd "$(dirname "$0")"
trap 'echo "FAILED: line $LINENO status $?"; exit 1' ERR

PY="${PY:-python3}"
unset V4_FEAT_DIR V4_SEED_SETS V4_ALL_LAYERS V4_LAYER_SUBSET V4_SCORES_DIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export V4_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

mkdir -p logs feats reef_acts sota_scores figs_sota

echo "V5_BEGIN $(date -Iseconds)"
"$PY" -u suite_sota.py status

list_missing() {
  local kind="$1"
  "$PY" - "$kind" << 'PY'
import sys
import suite_sota as S
kind = sys.argv[1]
locate = S.locate_ours_feat if kind == "ours" else S.locate_reef_feat
for k in S.SUITE_MODELS:
    if locate(k) is None:
        print(k)
PY
}

echo "--- collect ours ---"
mapfile -t OURS < <(list_missing ours)
if ((${#OURS[@]})); then
  for k in "${OURS[@]}"; do
    echo "OURS_BEGIN $k"
    "$PY" -u suite_sota.py collect_ours "$k"
    echo "OURS_END $k"
  done
else
  echo "OURS_NONE"
fi

echo "--- collect reef ---"
mapfile -t REEF < <(list_missing reef)
if ((${#REEF[@]})); then
  for k in "${REEF[@]}"; do
    echo "REEF_BEGIN $k"
    "$PY" -u suite_sota.py collect_reef "$k"
    echo "REEF_END $k"
  done
else
  echo "REEF_NONE"
fi

echo "--- run_static ---"
"$PY" -u suite_sota.py run_static all
echo "--- mechanism ---"
"$PY" -u suite_sota.py mechanism > sota_scores/mechanism_2x2_stdout.json
echo "--- evaluate ---"
"$PY" -u suite_sota.py evaluate
echo "--- plot ---"
"$PY" -u suite_sota.py plot
echo "V5_DONE $(date -Iseconds)"
