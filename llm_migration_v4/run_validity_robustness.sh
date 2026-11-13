#!/bin/bash
# Sequential CPU run for the 2026-11-11 validity/robustness plan.
# Skips caches that already exist. Stops on the first failure.
# Optional unrelated models are not in this script.
set -euo pipefail
cd "$(dirname "$0")"
unset V4_FEAT_DIR V4_SEED_SETS V4_ALL_LAYERS V4_LAYER_SUBSET
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export V4_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export V4_SCORES_DIR=scores_panel
export CUDA_VISIBLE_DEVICES=""
PY="${PY:-python3}"

run() {
  echo "START $(date -Is) $*"
  if ! "$PY" -u "$@"; then
    echo "FAILED $*"
    exit 1
  fi
  echo "OK $(date -Is) $*"
}

diffip() { run collect.py main "$1"; }

echo "PANEL_BEGIN $(date -Is)"
diffip pos_vicuna
run collect_reef_tqa.py victim fine_tune derivative_cn pos_vicuna \
  neg_qwen2_5_7b neg_llama3_8b neg_llama1_7b neg_baichuan_7b neg_mpt_7b
run collect_reef_tqa.py neg_pruned_sparsegpt neg_pruned_wanda neg_pruned_gblm

for name in \
  pos_sheared_1_3b_pruned pos_sheared_1_3b pos_sheared_1_3b_sharegpt \
  pos_sheared_2_7b_pruned pos_sheared_2_7b pos_sheared_2_7b_sharegpt \
  cpt_codellama cpt_wizardmath cpt_llemma
do
  diffip "$name"
done

run collect_reef_tqa.py \
  pos_sheared_1_3b_pruned pos_sheared_1_3b pos_sheared_1_3b_sharegpt \
  pos_sheared_2_7b_pruned pos_sheared_2_7b pos_sheared_2_7b_sharegpt \
  cpt_codellama cpt_wizardmath cpt_llemma

run score.py main \
  pos_vicuna \
  pos_sheared_1_3b_pruned pos_sheared_1_3b pos_sheared_1_3b_sharegpt \
  pos_sheared_2_7b_pruned pos_sheared_2_7b pos_sheared_2_7b_sharegpt \
  cpt_codellama cpt_wizardmath cpt_llemma
run score_reef_tqa.py
run exp_validity_tables.py
echo "PANEL_DONE $(date -Is)"
