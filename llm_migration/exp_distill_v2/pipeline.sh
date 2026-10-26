#!/bin/bash
set -e; cd /workspace/papers/exp_distill_v2; P=/workspace/.venv-fp/bin/python
while pgrep -f gen_teacher.py >/dev/null; do sleep 20; done
$P train_student.py distill 3 > logs/train_distill.log 2>&1
$P train_student.py control 3 > logs/train_control.log 2>&1
$P eval_strength.py > logs/eval_strength.log 2>&1
$P extract.py teacher student_distill student_control smollm135_base smollm360 gpt2 pythia160 qwen2_05b > logs/extract.log 2>&1
echo PIPELINE_DONE >> logs/extract.log
