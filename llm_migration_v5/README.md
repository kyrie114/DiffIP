# llm_migration_v5：论文缺口补全

v4 的 7B 主线、消融和判定入口保持原样。这一栏目只做 `论文判断.md` 里还不能写成主结论的四件事。规则在 [`试验方案.md`](试验方案.md)，跑之前已经写死，看到分数之后不改。

权重空间的非线性伪装这一轮不做。威胁模型仍是白盒隐藏状态。

## 四项

| 项 | 脚本 | 状态 |
|---|---|---|
| 同一朵扰动云上的线性 CKA，以及同维余弦 | `exp_gap_cka.py` | 已算完，`exp_gap_cka_result.json` |
| 新种子 3123 / 4123 / 5123 上的三条阈值 | `exp_gap_seeds.py` | 已算完，`exp_gap_seeds_result.json` |
| 官方 REEF 式干净激活 | `collect_reef.py`，`exp_gap_reef.py` | 已算完，`exp_gap_reef_result.json` |
| 第二个底座 Qwen3-8B | v4 `collect.py` 写入 `feats_qwen3/`，再跑 `exp_gap_qwen3.py` | 已算完，12 对都不通过 |

数字和通过与否写在 [`进度.md`](进度.md)。历史均匀 4 层特征仍用 v4 的 `feats/` 与 `feats_all/`，本目录不覆盖它们。

## 怎么跑

评分与采集沿用 v4 的环境：conda 环境 `diffip-llm`，`HF_HUB_OFFLINE=1`，CPU，`V4_THREADS=8`。

干净激活在本目录：

```bash
python -u collect_reef.py
python -u exp_gap_reef.py
```

第二个底座在 v4 目录采集，输出到本目录，种子仍是原来的三组：

```bash
V4_FEAT_DIR=../llm_migration_v5/feats_qwen3 V4_SEED_SETS=123,1123,2123 \
  python collect.py main qwen3_victim qwen3_instruct qwen3_hard
```

然后回到本目录跑 `python -u exp_gap_qwen3.py`。
