# llm_migration_v3 — DiffIP→LLM 严格重跑

按 DiffIP 原文做大数定律采样：32 个 prompt、3 个 seed set、N 扫到 128；主指标是 ALS（QSX−h）平均 L2 → `exp(−mean L2)`，**不用 margin**。含解码对照、剪枝 DP 对照、以及同模型对上的 REEF/CKA 基线（`reef_baseline/`）。

入口说明见 [`docs/LLM迁移实验导读.md`](../docs/LLM迁移实验导读.md)；详细数字与分析见 [`实验报告.md`](实验报告.md)。复现顺序：`collect_perturb.py` → `score.py pert` → `aggregate.py`（可选 `score_dp.py` / `collect_decode.py`）。
