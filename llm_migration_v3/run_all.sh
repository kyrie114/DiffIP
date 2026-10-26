#!/bin/bash
# resumable pipeline; every stage skips finished per-model / per-pair files
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1; PY=/workspace/.venv-fp/bin/python
$PY collect_perturb.py && $PY score.py pert && $PY score_dp.py && \
$PY collect_decode.py && $PY score.py dec && echo ALL_DONE
