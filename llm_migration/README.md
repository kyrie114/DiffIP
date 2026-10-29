# DiffIP → LLM 迁移实验（本 fork 附加内容）

本目录是在官方 [zhuolingli/DiffIP](https://github.com/zhuolingli/DiffIP) 之上，把 **DiffIP 的 `QSX-h` 回拉 / PCA / DP** 迁到大语言模型（LLM）隐藏层表征上的实验记录。  
**上游 `diffip/` 包未改动**；可复用本仓库根目录的 `diffip`（`DiffIPMetric`）。

> 原则：不编造数字；负结果如实写。大文件（权重、缓存特征）不进仓库，只留在本地。

> **定位**：本目录是最早一版实验（主判定曾用 margin，已否）。权威导读与两台机器方法对照见 [`docs/LLM迁移实验导读.md`](../docs/LLM迁移实验导读.md)；当前主结论在 [`llm_migration_v3/`](../llm_migration_v3/)。


## 总览结论（诚实）

| 实验 | 问题 | 结论 |
|---|---|---|
| [exp_methodA_diffip20](exp_methodA_diffip20/) | 官方 DiffIPMetric 直接打 LLM 逐层 hidden，能否当指纹？ | S1 微调可分，但 **margin 远小于直接余弦**；全维回拉几乎饱和；弱蒸馏学生加入后全部失败 |
| [exp_whitebox_compare](exp_whitebox_compare/) | DiffIP-20 vs REEF(CKA) vs 直接余弦 | **微调线上 DiffIP 与 REEF 打平，余弦更强**；DiffIP ≈ 30× REEF 耗时 |
| [exp_rotation_attack](exp_rotation_attack/) | 保函数正交旋转/置换后谁还活着？ | 余弦崩掉；**DiffIP 不优于只中心化的 CKA**；掉分来自 RMSNorm 折叠而非旋转 |
| [exp_distill_v2](exp_distill_v2/) | 更充分的异架构蒸馏，hidden 指纹能否认出学生？ | **负结果**：行为上已靠近教师，但 DiffIP / CKA 都认不出（与基座、对照学生差 < 0.003） |

**对「微调线」的判断**：DiffIP 回拉在数学上可用，但解决的「坐标系被变换」问题，CKA 用更低成本已覆盖；微调本身几乎不换坐标系，直接余弦最灵敏。  
**对「蒸馏线」的判断**：序列级异词表蒸馏主要改输出行为，不改预训练留下的 hidden 几何；基于 hidden 的指纹（含 DiffIP）对这类蒸馏不敏感。

## 目录结构

```
llm_migration/
├── README.md                 ← 本文件
├── figs/                     ← 汇总图
├── exp_methodA_diffip20/     ← 方法 A：官方 DiffIP → LLM
├── exp_whitebox_compare/     ← 与 REEF / 余弦对照
├── exp_rotation_attack/      ← 保函数旋转/置换攻击
└── exp_distill_v2/           ← 蒸馏重做（异架构学生）
```

未入库（体积大，留在实验机）：
- `exp_rotation_attack/feats/`（`.npy` 隐藏特征）
- `exp_distill_v2/ckpt/`（两个学生各约 517MB）
- `exp_distill_v2/feats/`
- 更早的特征缓存：`/workspace/20261019_DiffIP迁移LLM_线性回拉/feats`、`/workspace/reef_traj/feats`（仅实验机路径）

## 与官方 DiffIP 的对应关系

- 入口：仓库根目录 [`diffip/metric.py`](../diffip/metric.py) → `DiffIPMetric`
- LLM 迁移时的轴替换（各实验 `steps()` / 报告中有说明）：
  - 扩散 **时间步** → Transformer **层**（等距取 7 层）
  - 扩散 **多种子** → **多条探测 Prompt**（同提示跨模型配对）
  - 额外做了样本维去均值 + 均方范数归一化（避免 `exp(-d)` 下溢）；**官方 README 无此步**，报告中已标明偏离

依赖：`numpy` / `scipy` / `scikit-learn`；跑抽取脚本还需要 `torch` + `transformers` 与对应模型权重。打分阶段若已有 `.npy`，只需官方 `diffip` 包。

---

## 1. exp_methodA_diffip20

**做什么**：把官方 `DiffIPMetric(n_components=K)` 打在真实 LLM 逐层 hidden 上（S1 微调设定 + S2 含弱蒸馏学生）。

**关键数字**（详见 [`实验报告.md`](exp_methodA_diffip20/实验报告.md)、[`results.json`](exp_methodA_diffip20/results.json)）：

- S1（Qwen2.5-0.5B；正：Instruct / LoRA；负：Qwen2-0.5B）  
  - 直接余弦 margin **+0.72**；DiffIP pca20 **+0.025**；全维 **+0.0033**
- S2 不含学生：pca20 可分 6/6，margin 仅 **+0.007**  
- S2 含弱 GPT2 学生：所有 K，可分子集 **0/6**

**如何跑**（需本机已有特征缓存，路径写在 `run.py` 里）：

```bash
# 在仓库根目录，PYTHONPATH 含本仓库
python llm_migration/exp_methodA_diffip20/run.py
python llm_migration/exp_methodA_diffip20/analyze.py
python llm_migration/exp_methodA_diffip20/plot.py
```

图：[`figs/methodA_diffip20_margin.png`](figs/methodA_diffip20_margin.png)

---

## 2. exp_whitebox_compare

**做什么**：同一特征与子集上对比 REEF（用户口径 lin_cka）、直接余弦、DiffIP-20。

**关键数字**（[`实验报告.md`](exp_whitebox_compare/实验报告.md)）：

- S1：三者 AUC 皆 1；REEF margin +0.027 ≈ DiffIP +0.025 ≪ 余弦 +0.72  
- S2 无学生：DiffIP 6/6 可分 vs REEF 2/6，但 DiffIP margin 仅 +0.007，原因未拆清  
- DiffIP 耗时约 REEF 的 **30×**  
- 含学生后三种方法均失败

```bash
python llm_migration/exp_whitebox_compare/compare.py
python llm_migration/exp_whitebox_compare/plot.py
```

图：[`figs/whitebox_compare_margin.png`](figs/whitebox_compare_margin.png)

---

## 3. exp_rotation_attack

**做什么**：对 Instruct 衍生模型做保函数的维度置换 / 正交旋转（SliceGPT/QuaRot 式 RMSNorm 折叠），验证余弦 / CKA / DiffIP 谁还活着。

**关键数字**（[`实验报告.md`](exp_rotation_attack/实验报告.md)、[`verify.json`](exp_rotation_attack/verify.json)）：

- 函数保持：logits 最大差 ~1e-4；贪心生成 10/10 相同  
- 攻击后余弦 → −0.01～−0.03（AUC 0.5）  
- 只中心化 CKA 在旋转下仍 **0.995**；带标准化的 REEF 口径旋转后 **0.962**  
- DiffIP：0.996 → 0.989，**与 fold-only 对照相同**，掉分来自折叠不是旋转  
- 攻击后 margin：DiffIP +0.037 vs 只中心化 CKA +0.052 → **方法 A 不优于 CKA**

```bash
# 需要模型权重；feats/ 默认写到本目录（未入库）
python llm_migration/exp_rotation_attack/extract.py
python llm_migration/exp_rotation_attack/score.py
python llm_migration/exp_rotation_attack/plot.py
```

图：[`figs/rotation_attack_scores.png`](figs/rotation_attack_scores.png)

---

## 4. exp_distill_v2

**做什么**：重做蒸馏线——SmolLM2-135M 对学生，Qwen2.5-0.5B-Instruct 为教师，Alpaca 上序列级 KD；对照学生用 Alpaca 原答案。

**关键数字**（[`实验报告.md`](exp_distill_v2/实验报告.md)、[`strength.json`](exp_distill_v2/strength.json)、[`results.json`](exp_distill_v2/results.json)）：

- 蒸馏强度（留出）：教师回答 NLL/字节 0.348→**0.256**；top-1 0.601→**0.675**；ROUGE-L 0.123→**0.260**（对照学生更差或更偏 Alpaca）  
- 指纹 margin（对基座+对照）：DiffIP20 **−0.0003**；CKA_plain **−0.004**；CKA_std **−0.002**  
- 同家族 Qwen2-0.5B 的 DiffIP（0.952）反高于蒸馏学生（0.918）

**未入库**：`ckpt/`（学生权重）、`feats/`。`data/teacher_data.json`（约 1.5MB）已入库便于复现生成侧。

```bash
bash llm_migration/exp_distill_v2/pipeline.sh   # 全流程；需 GPU/CPU 与模型
# 或分步：gen_teacher → train_student → eval_strength → extract → score → plot
```

图：[`exp_distill_v2/distill_v2.png`](exp_distill_v2/distill_v2.png)

---

## 作者与提交说明

- 实验与整理：陆顺鑫 `<2018869035@qq.com>`
- 本目录为 fork 附加实验，**不覆盖**上游 DiffIP 实现
- 报告中的绝对路径（如 `/workspace/...`）是原实验机路径，换机器需改脚本里的特征/模型路径


> 协议已纠正：请看 [`llm_migration_v2/`](../llm_migration_v2/)（多样本 + ALS mean L2，无 margin 主表）。本目录保留为历史对照。
