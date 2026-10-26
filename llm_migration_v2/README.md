# llm_migration_v2 — DiffIP-faithful LLM 迁移（ALS only）

对齐官方 DiffIP：`LLMRepresentationCollector` 多样本扰动 + `reversion.fit_step_alignment`（QSX-h ALS）mean L2。  
**不用 margin 做主指标；不用 DP。**

## 跑法

```bash
cd /path/to/DiffIP
PYTHONPATH=. python llm_migration_v2/run.py
```

依赖：`torch`、`transformers`、`scikit-learn`、`scipy`（与官方 DiffIP 相同）。

## 文件

- `run.py` — 采集 + 四条线评分
- `score_als.py` — PCA + 逐层 ALS（维数见 pca_sweep），无 DP
- `transforms.py` — 置换 / 剪枝（缩放在表征上做对角 S）
- `collect_shared.py` — 权重伪装时的共享 embedding 云
- `results.json` — 原始数字
- `实验报告.md` — 结论
- `PROTOCOL.md` — 与旧 `llm_migration/` 的差异
- `pca_sweep/` — PCA 维数网格（见 [`pca_sweep/维数扫描.md`](pca_sweep/维数扫描.md)）；推荐 **8**，勿盲用 20
