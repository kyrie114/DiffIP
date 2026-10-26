# llm_migration_v2 vs 旧 llm_migration：协议对照

## 旧 llm_migration（被批评为偏离 DiffIP）

| 项 | 旧做法 |
|---|---|
| 采样 | 多条真实 prompt，各跑一遍；取 **last-token** hidden；无 embedding 扰动，无配对种子 |
| 大数定律 | 未用 DiffIP 的多种子/多样本扰动云 |
| 主指标 | 自造 **margin** / AUC（正样本最低分 − 负样本最高分） |
| 回拉 | 有时手写 ALS，有时与官方 `DiffIPMetric` 混用；曾加 scale 归一化 |
| DP | 参与或不清 |
| 范围 | 微调 + 旋转 + **蒸馏** 等 |

## 本目录 llm_migration_v2（按官方 / 你的 collector）

| 项 | 本协议 |
|---|---|
| 采样 | `LLMRepresentationCollector`，**同一 prompt**，N=8，`perturbation_scale=2e-3`，**seed=123 配对** |
| 池化 | attention-mask **mean pool**（与 `diffip/llm.py` 一致） |
| 形状 | `[L, N, D]`，L=4 均匀层 |
| 主指标 | 官方 `fit_step_alignment` 的 **mean L2**（`distance = ‖QSX−h‖² / N`），再对层取平均；`similarity = exp(−∑d_ℓ / L)` |
| DP | **关闭**：等长层 1:1 配对；剪枝用 matched-subset，并在结果里注明「不是 DP」 |
| 范围 | 仅 A 微调 / B 置换 / C 各向异性缩放 / D 剪枝 |
| 不做 | margin 主表、蒸馏线、AUC 主结论 |

## 置换实验的额外注意

权重置换后若对 **各自** 的 `embed_tokens` 加独立噪声，N 个样本不再整体差一个置换矩阵。  
本实验对置换模型喂 `(E+ε)@P`（与干净侧共享 ε），保证表征云满足 `H_perm = H @ P`，再交给 ALS。
