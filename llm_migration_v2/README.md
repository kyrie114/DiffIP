# llm_migration_v2 — 修正协议（ALS + 平均 L2）

对齐 DiffIP：配对 embedding 扰动 + `reversion.fit_step_alignment`（QSX−h）的 **mean L2**；`similarity = exp(−mean L2)`。**不用 margin 做主指标；DP 默认关。** 单 prompt、N=8；PCA 扫描推荐 **8 维**（见 `pca_sweep/`）。

本目录是「严格重跑」之前的修正版，主结论请看 [`llm_migration_v3/`](../llm_migration_v3/)。总览与两台机器方法对照：[`docs/LLM迁移实验导读.md`](../docs/LLM迁移实验导读.md)。

```bash
PYTHONPATH=. python llm_migration_v2/run.py
```
