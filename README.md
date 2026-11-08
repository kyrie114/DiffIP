# DiffIP: Representation Fingerprinting for Diffusion Models & Large Language Models

本项目基于 DiffIP 理论框架，研究将深度生成模型（Diffusion）的表示对齐与几何指纹扩展迁移至大语言模型（LLM）的谱系溯源与版权检测。

---

## 目录索引与架构说明

| 模块目录 | 核心定位与说明 |
|---|---|
| `diffip/` | DiffIP 核心算法库（含 Procrustes、ALS 回退对齐算法与表示采集器） |
| `llm_migration_v4/` | **当前主线研究目录（7B 规模验证与消融实验）**：包含完整 Llama-2-7B 真实谱系验证、三大消融实验、五无关模型扩盘及精美汇报 PPT 产物 |
| `llm_migration_v3/` | 早期 0.5B (Qwen2.5) 扰动验证目录（探索性阶段） |
| `llm_migration_v2/` | 协议修正与早期单 Prompt 探索（历史归档） |
| `llm_migration/` | 最早原型版本（仅供历史参考，已被 v3/v4 协议替代） |
| `llm_transfer/` | 另一实验机探索代码（针对小模型探索工作点） |
| `docs/` | 整体实验导读文档与理论备忘录 |
| `notes/` | 指纹方法与技术路线调研笔记 |

---

## 主线研究入口 (v4 阶段)

请进入 [`llm_migration_v4/`](llm_migration_v4/) 目录：
- 完整实验结果汇总：[`llm_migration_v4/实验汇报_主实验与消融_20261108.md`](llm_migration_v4/实验汇报_主实验与消融_20261108.md)
- 决策代码与判定接口：[`llm_migration_v4/lineage.py`](llm_migration_v4/lineage.py)
- 面向导师汇报的精编 PPT：[`llm_migration_v4/汇报_DiffIP_大模块与消融实验_v2.pptx`](llm_migration_v4/汇报_DiffIP_大模块与消融实验_v2.pptx)
