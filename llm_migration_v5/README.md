# llm_migration_v5

这个目录里有两套已经分开的材料。

## 论文缺口补全

v4 的 7B 主线、消融和判定入口保持原样。这一部分只做 `论文判断.md` 里还不能写成主结论的四件事。规则在 [`试验方案.md`](试验方案.md)，跑之前已经写死，看到分数之后不改。

权重空间的非线性伪装这一部分不做。威胁模型仍是白盒隐藏状态。

| 项 | 脚本 | 状态 |
|---|---|---|
| 同一朵扰动云上的线性 CKA，以及同维余弦 | `exp_gap_cka.py` | 已算完，`exp_gap_cka_result.json` |
| 新种子 3123 / 4123 / 5123 上的三条阈值 | `exp_gap_seeds.py` | 已算完，`exp_gap_seeds_result.json` |
| 官方 REEF 式干净激活 | `collect_reef.py`，`exp_gap_reef.py` | 已算完，`exp_gap_reef_result.json` |
| 第二个底座 Qwen3-8B | v4 `collect.py` 写入 `feats_qwen3/`，再跑 `exp_gap_qwen3.py` | 已算完，12 对都不通过 |

数字和通过与否写在 [`进度.md`](进度.md)。历史均匀 4 层特征仍用 v4 的 `feats/` 与 `feats_all/`，这一部分不覆盖它们。

评分与采集沿用 v4 的环境：conda 环境 `diffip-llm`，`HF_HUB_OFFLINE=1`，CPU，`V4_THREADS=8`。

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

## 论文投稿实验总套件

本目录也放面向论文撰写的补充 SOTA 对比、计算效率对比、机理拆解图表与方案文档，与前期探索性的 `llm_migration_v4/` 分开。

`suite_sota.py` 以只读方式复用 `../llm_migration_v4/` 中已采集的 `.npz` 特征与已跑完的消融 JSON。新产生的特征、分数、Markdown/LaTeX 主表和论文图表写在本目录内（`feats/`、`reef_acts/`、`sota_scores/`、`figs_sota/`）。

- `suite_sota.py`：一键实验执行与出图脚本（`status` / `collect_ours` / `collect_reef` / `run_static` / `mechanism` / `evaluate` / `plot`）
- `二合一完整实验设计方案_广度4x4与深度机理.md`：主方案
- `试验方案_导师手稿4基座4衍生与SOTA对比.md`：导师手稿 4 基座 × 4 衍生类型专项方案
- `导师手稿对应实验落地指南.md`：导师手稿逐块解析与度量指标说明
- `论文实验章节完整设计方案.md`：单主基座深挖与机理拆解初版方案
- `figs_sota/`：论文图

```bash
cd llm_migration_v5
export HF_HUB_OFFLINE=1
export V4_THREADS=16

python suite_sota.py status
python suite_sota.py collect_ours all
python suite_sota.py collect_reef all
python suite_sota.py run_static all
python suite_sota.py mechanism
python suite_sota.py evaluate
python suite_sota.py plot
```
