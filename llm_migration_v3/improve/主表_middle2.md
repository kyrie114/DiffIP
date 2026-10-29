# 主表：all4 → middle2（v3 扰动面板重算）

日期：2026-10-29（Asia/Shanghai）。  
脚本：`llm_migration_v3/score_middle2.py`。  
产物：`tables_pert_middle2.md`、`summary_pert_middle2.json`。  
**不重新采集**：复用 `feats/pert_*.npz`；middle2 = 4 均匀层索引 `[1, 2]`（与 `exp2_prune_layers.py` 一致）。

## 协议差异

| 项 | all4（原主表） | middle2（本表） |
|---|---|---|
| 层 | `[0,1,2,3]`（相对深度均匀） | `[1,2]` |
| 相似度 | `exp(−Σ d_ℓ / L)`，L=4 | 同式，L′=2 |
| 秩约束 | `dims ≤ 4N−1` | `dims ≤ 2N−1`（故 N=8 不下 dims=20） |
| PCA / ALS / DP | 官方 project_sequences + fit_step_alignment，DP off | 相同 |

## 检测口径对照（seed set 0，N=128，dims=8，32 prompts）

| 设定 | gap = min(derived)−max(neg) | 合并 AUC | wins（各衍生 vs 5 负例） |
|---|---:|---:|---|
| all4 | 0.411 | 0.953 | 全部 32/32 |
| **middle2** | **0.605** | **0.989** | 全部 **32/32** |

逐对 mean similarity：

| pair | all4 | middle2 | Δ |
|---|---:|---:|---:|
| fine_tune | 0.758 | 0.766 | +0.008 |
| weight_perm | 0.758 | 0.766 | +0.008 |
| rep_perm | 0.758 | 0.766 | +0.008 |
| scale | 0.753 | 0.765 | +0.012 |
| prune_dpoff | 0.646 | **0.749** | **+0.103** |
| neg_smollm2_135m | 0.235 | 0.144 | −0.091 |
| neg_smollm2_360m | 0.235 | 0.144 | −0.091 |
| neg_gpt2 | 0.234 | 0.144 | −0.090 |
| neg_pythia_160m | 0.235 | 0.144 | −0.091 |
| neg_qwen2_0.5b | 0.234 | 0.143 | −0.091 |

要点：
1. **剪枝提升最大**（0.646→0.749）；负例整体下压约 0.09，gap 净增约 **+0.19**。
2. 微调 / 置换 / 缩放几乎不动（+0.01 量级），说明 middle2 不是「全面抬分」，而是去掉对剪枝有害的浅层/末层。
3. 与 improve 试探（仅 seed0、N=128）一致：gap≈0.60、AUC≈0.99；本表补全了 3 seed-set 与 N 网格，可作为 **剪枝友好的改进默认层**。
4. **all4 数字仍作基线**写在 `tables_pert.md` / `实验报告.md`；汇报准确率时同时报 middle2。

## 3 seed-set 收敛（dims=8，sim mean±std）

| pair | N=20 all4 | N=20 middle2 | N=128 all4 | N=128 middle2 |
|---|---:|---:|---:|---:|
| fine_tune | 0.792±0.004 | 0.811±0.005 | 0.760±0.005 | 0.766±0.006 |
| prune_dpoff | 0.690±0.004 | 0.793±0.004 | 0.645±0.001 | 0.749±0.006 |
| max(5 neg) | ≈0.328 | ≈0.249 | ≈0.238 | ≈0.148 |

完整格点见 [`tables_pert_middle2.md`](../tables_pert_middle2.md)。
