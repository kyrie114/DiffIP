#!/bin/sh
# 每个模型单独一个进程（内存紧），meta.json 里已有的跳过；中断后重跑本脚本即可续上。
for n in instruct_wperm instruct_prune neg_qwen2_0.5b neg_smollm2_135m drop2 drop6 drop8 drop12 width75 width50 width75h width50h vocab_ext vocab_swap; do
  python3 sa_collect.py $n 2>&1 | grep -v -E "Loading|Warning" >> collect.log
done
echo ALLDONE >> collect.log
