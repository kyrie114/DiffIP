# `llm_migration_v5`: 论文投稿实验总套件（二合一：4×4 跨基座 SOTA 对比 + 单基座低误报深挖与机理拆解）

> **目录隔离设计**：本目录专门用于存放面向论文最终撰写的**补充 SOTA 对比、计算效率对比、机理拆解图表与方案文档**，与前期探索性的 `llm_migration_v4/` 严格分离、互不混杂。
> [suite_sota.py](file:///d:/study/Diffip/llm_migration_v5/suite_sota.py) 会**以只读方式**自动复用 `../llm_migration_v4/` 中已采集的 `.npz` 特征与已跑完的消融 JSON 结果，所有新产生的特征、分数、Markdown/LaTeX 主表和论文图表均只写入本目录内（`feats/`、`reef_acts/`、`sota_scores/`、`figs_sota/`）。

## 目录文件说明

- [suite_sota.py](file:///d:/study/Diffip/llm_migration_v5/suite_sota.py)：一键实验执行与出图脚本（支持 `status` / `collect_ours` / `collect_reef` / `run_static` / `mechanism` / `evaluate` / `plot`）
- [二合一完整实验设计方案_广度4x4与深度机理.md](file:///d:/study/Diffip/llm_migration_v5/二合一完整实验设计方案_广度4x4与深度机理.md)：**最新主方案**（整合导师 4×4 广度骨架 + 单基座深挖打 REEF 误报 + 3 步创新原理拆解 + 8 张实验表规划）
- [试验方案_导师手稿4基座4衍生与SOTA对比.md](file:///d:/study/Diffip/llm_migration_v5/试验方案_导师手稿4基座4衍生与SOTA对比.md)：导师手稿 4 基座 × 4 衍生类型专项方案
- [导师手稿对应实验落地指南.md](file:///d:/study/Diffip/llm_migration_v5/导师手稿对应实验落地指南.md)：导师手稿逐块解析与度量指标说明
- [论文实验章节完整设计方案.md](file:///d:/study/Diffip/llm_migration_v5/论文实验章节完整设计方案.md)：单主基座深挖与机理拆解初版方案（已并入二合一主方案）
- [figs_sota/](file:///d:/study/Diffip/llm_migration_v5/figs_sota/)：5 张论文核心图（`.png` + Overleaf `.pdf`）

## 服务器一键运行

```bash
cd /path/to/DiffIP/llm_migration_v5
export HF_HUB_OFFLINE=1
export V4_MODELS=/data/luo_lixuan/llms
export V4_THREADS=16

python suite_sota.py status
python suite_sota.py collect_ours all
python suite_sota.py collect_reef all
python suite_sota.py run_static all
python suite_sota.py mechanism
python suite_sota.py evaluate
python suite_sota.py plot
```
