#!/bin/bash
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1; PY=/workspace/.venv-fp/bin/python
$PY collect_decode.py && $PY score.py dec && $PY aggregate.py dec && $PY diagnose.py && echo ALL_DONE
