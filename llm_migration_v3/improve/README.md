# improve/ — 提高 DiffIP→LLM 检测准确率的跟进实验

依据协作方优先级（2026-10-29）：
1. **公平复测**：仅旋转+对角缩放+平移（Q、S、h）的表征回拉 vs 直接余弦；加大校准样本量。
2. **剪枝专项**：层选择（中层 / 末 token）；**已落地主表** `../tables_pert_middle2.md`。
3. **蒸馏/抽取鲁棒性**：`exp3_distill_robust.py` + `蒸馏鲁棒性.md`（既有 student ckpt，CPU 代理；负结果）。
4. CKA 融合次要；LoRA / 更大模型不当主线。

脚本与报告：
- `exp1_qsh_vs_cosine.py` / `results_exp1*.json` → `提升报告.md` §1
- `exp2_prune_layers.py` / `results_exp2.json` → `提升报告.md` §2
- `../score_middle2.py` → `主表_middle2.md`、`../tables_pert_middle2.md`
- `exp3_distill_robust.py` / `results_exp3_distill.json` → `蒸馏鲁棒性.md`

数字全部实跑；负结果如实写。大文件 `feats_distill/` 不入库。
