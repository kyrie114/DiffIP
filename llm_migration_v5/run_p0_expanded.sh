#!/usr/bin/env bash
# P0 expanded pool runner: FuseLLM-7B / Abel-7B-002 + 10 diverse 7B negatives (12 models),
# optional --include-large adds Mixtral-8x7B-Instruct-v0.1 & internlm2-chat-20b-llama-old.
# Runs one model per fresh process, skips already-collected npz caches, isolates per-model failures.
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
EXTRA_FLAG=""
if [[ "${1:-}" == "--include-large" ]]; then
  EXTRA_FLAG="--include-large"
fi

unset V4_FEAT_DIR V4_SEED_SETS V4_ALL_LAYERS V4_LAYER_SUBSET V4_SCORES_DIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export V4_THREADS="${V4_THREADS:-8}"
export OMP_NUM_THREADS="${V4_THREADS}" MKL_NUM_THREADS="${V4_THREADS}" OPENBLAS_NUM_THREADS="${V4_THREADS}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

mkdir -p logs feats reef_acts sota_scores docs

echo "P0_BEGIN $(date -Iseconds) include_large=${EXTRA_FLAG:-false}"
"$PY" -u exp_p0_expanded_pool.py status $EXTRA_FLAG

KEYS=($("$PY" -c "import exp_p0_expanded_pool as P; print(' '.join(P.active_keys(${EXTRA_FLAG:+True})))"))

echo "--- Step 1: collect Ours (even4 + PCA-8) ---"
FAILED_OURS=()
for k in "${KEYS[@]}"; do
  echo "P0_OURS_BEGIN $k $(date -Iseconds)"
  if ! "$PY" -u exp_p0_expanded_pool.py collect_ours "$k" $EXTRA_FLAG 2>&1 | tee "logs/p0_ours_${k}.log"; then
    echo "WARN: Ours failed on $k (check logs/p0_ours_${k}.log), continuing..."
    FAILED_OURS+=("$k")
  fi
done

echo "--- Step 2: collect REEF (TruthfulQA-200) ---"
FAILED_REEF=()
for k in "${KEYS[@]}"; do
  echo "P0_REEF_BEGIN $k $(date -Iseconds)"
  if ! "$PY" -u exp_p0_expanded_pool.py collect_reef "$k" $EXTRA_FLAG 2>&1 | tee "logs/p0_reef_${k}.log"; then
    echo "WARN: REEF failed on $k (check logs/p0_reef_${k}.log), continuing..."
    FAILED_REEF+=("$k")
  fi
done

echo "--- Step 3: evaluate & generate summary ---"
"$PY" -u exp_p0_expanded_pool.py evaluate $EXTRA_FLAG | tee logs/p0_evaluate.log

echo "P0_DONE $(date -Iseconds)"
echo "Failed Ours: ${FAILED_OURS[*]:-none}"
echo "Failed REEF: ${FAILED_REEF[*]:-none}"
