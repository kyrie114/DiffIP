# DiffIP → LLM 迁移实验导读

面向：需要搞清「仓库里干了什么、两台机器的方法有何区别」的同学。  
**本文件是 DiffIP→LLM 实验的权威入口**；旧笔记 [`notes/LLM指纹方法对比与实验笔记.md`](../notes/LLM指纹方法对比与实验笔记.md) 里混入了别的指纹讨论，以本导读为准。

原则：不编造数字；负结果如实写。大文件（权重、`feats/`、`acts/`、日志）不进仓库。

---

## 1. 仓库现在怎么走（目录地图）

```
DiffIP/                          ← fork: kyrie114/DiffIP（上游实现未改）
├── diffip/                      ← 官方包（DiffIPMetric、reversion、llm 采集器）
├── docs/LLM迁移实验导读.md      ← 你正在读的文件
├── llm_transfer/                ← 【另一台机 / 用户线】collector + DiffIPMetric
├── llm_migration/               ← 【最早版本】已被否掉，只作历史对照
├── llm_migration_v2/            ← 【修正协议】ALS + 平均 L2；单 prompt、N=8
├── llm_migration_v3/            ← 【严格重跑】主结论所在；含 decode / DP / REEF / improve
└── notes/                       ← 杂记；DiffIP→LLM 请回本导读
```

| 文件夹 | 谁做的 | 角色 | 先读什么 |
|---|---|---|---|
| `llm_transfer/` | 另一台机（kyrie114） | 探索：扰动尺度、真实 SFT 对、消融、ROC | [`REPORT.md`](../llm_transfer/REPORT.md) |
| `llm_migration/` | 本 bot 最早一版 | **已否**：用 margin 当主判定，偏离 DiffIP | [`README.md`](../llm_migration/README.md) |
| `llm_migration_v2/` | 本 bot | 按 DiffIP 改回 ALS+mean L2；单提示、N=8；PCA→8 维 | [`实验报告.md`](../llm_migration_v2/实验报告.md) |
| `llm_migration_v3/` | 本 bot | **严格重跑**：32 提示、N→128、负例、decode、REEF | [`实验报告.md`](../llm_migration_v3/实验报告.md) |

口头上可以说「最早版本 / 修正协议 / 严格重跑」，路径仍用上面四个文件夹名。不要把「v1/v2/v3」当成论文里的正式命名。

---

## 2. 版本演进（一个版本一个版本讲）

按时间顺序，大白话说明每一版在干什么、为什么留下或否掉。

---

### 第 0 版：另一台机子的探索（`llm_transfer/`）

你在另一台电脑上先做的。做法是：用 collector 抓各层 hidden，再调用 DiffIP 自带的打分函数 `DiffIPMetric.compare`。它定下了后面大家都沿用的工作点：**同一份 embedding 噪声、尺度 2e-3、配对种子**。

主要结果：SmolLM2 基座对 Instruct（真实微调）相似度 **0.505**，跨家族只有 **0.323**，方向对；ROC 的 AUC 是 **0.678**。混合采样和把 N 加到 16 都会塌掉，所以这些路被划掉了。报告在 [`llm_transfer/REPORT.md`](../llm_transfer/REPORT.md)。

---

### 第 1 版：最早的迁移实验（`llm_migration/`）——已被否掉

我最早做的一版。问题有两个：一是用 **margin**（正负例分数差）当最终判据，不是 DiffIP 原文的「平均 L2 距离」；二是没有按大数定律做多样本采集。所以它和 DiffIP 原文对不上，后来被否掉。里面还有 methodA、白盒对比、旋转攻击、蒸馏那些子实验，都是这一套错误协议下的产物，只能当历史记录。

---

### 第 2 版：修正协议（`llm_migration_v2/`）

按要求改回 DiffIP 原文：PCA 降维 → ALS 拟合 Q、S、h → 用平均 L2 算相似度，去掉 margin，DP 先不用。还补了 PCA 维数扫描（选出 **8 维**）和 5 个无关模型的对比。

这一版能分开衍生模型和无关模型了，但还不严谨：只用了一个提示词「The quick brown fox…」，N 只有 8，谈不上大数定律。详见 [`llm_migration_v2/实验报告.md`](../llm_migration_v2/实验报告.md)。

---

### 第 3 版：严格重跑（`llm_migration_v3/`）——现在的主结果

「按这样再来」之后的版本。32 个提示词、3 组种子，N 从 8 扫到 128，同样 5 个无关模型。

核心数字（N=128）：微调 / 置换 / 缩放大约 **0.75–0.76**，剪枝 **0.645**，无关模型大约 **0.24**，32 个提示词全都分对。N 太小时无关模型也会虚高；N≥64 以后结果稳定。剪枝开不开 DP 几乎没差别。

同文件夹里还有两个对照：

- **解码采样**：让模型自己随机写字当样本，分数全掉到 0，证明必须用配对扰动。
- **REEF（CKA）**：按官方设置在这些小模型上分不开；但把 CKA 套在同一批扰动样本上，又能分得很开。说明真正起作用的主要是配对采样，ALS 是不是比 CKA 更强还说不准。

详细报告：[`llm_migration_v3/实验报告.md`](../llm_migration_v3/实验报告.md)；REEF 说明在 [`reef_baseline/说明.md`](../llm_migration_v3/reef_baseline/说明.md)。

---

### 提升跟进（`llm_migration_v3/improve/`，2026-10-29）

按协作方优先级做了两件事（报告：[`提升报告.md`](../llm_migration_v3/improve/提升报告.md)）：

1. **公平复测 QSh vs 余弦**：复现 10/19「余弦 AUC=1.0 / 全维 QSh 留出≈0.52」；但在 v3 扰动云、校准向量 M≥1024 时，全维 QSh 留出 AUC→1.0（能抗置换），余弦在含置换时只有 0.60。
2. **剪枝层选择**：v3 默认 4 均匀层改用 **middle2** 后，DiffIP gap 0.394→**0.601**，AUC 0.948→**0.988**。末层最差。

蒸馏线：未新开训练（CPU）；用既有 student ckpt 做 middle2 配对扰动重测，见 [`蒸馏鲁棒性.md`](../llm_migration_v3/improve/蒸馏鲁棒性.md)——相对教师仍 AUC≈0.5；相对基座认得出 SFT，但 distill 不优于 control。LoRA/更大模型跳过。

**改进默认层（middle2）**：在不重新采特征的前提下，把打分层从 4 均匀层改为中间两层索引 `[1,2]` 后，扰动面板（N=128, dims=8）gap **0.411→0.605**、合并 AUC **0.953→0.989**，剪枝相似度 0.646→0.749；微调/置换/缩放几乎不变。完整主表见 [`tables_pert_middle2.md`](../llm_migration_v3/tables_pert_middle2.md) 与 [`主表_middle2.md`](../llm_migration_v3/improve/主表_middle2.md)。**all4 数字仍保留为基线**（`tables_pert.md`）。

### FingerInv Table-2 攻击补测（`llm_migration_v3/fingerinv_attacks/`，2026-10-30）

对照 FingerInv（CVPR25）Table 2 的 **3 类攻击 / 22 格**（微调、剪枝 PN%、float16/bfloat16），在 v3 同一 LLM 面板上补强度梯度与量化变体。报告：[`FingerInv攻击补测.md`](../llm_migration_v3/fingerinv_attacks/FingerInv攻击补测.md)；数字：`results.json`。

middle2（N=128, dims=8）要点：**fp16≈微调**（sim 0.763 / gap +0.619）；**bf16 较弱**（0.580）；**真动态 int8 很弱**（0.195）；层丢弃到 30% 仍稳、**50% 崩**；幅值剪枝 30%/50% 明显掉。另有 **模拟** RTN-8/4（`q_w8sim`/`q_w4sim`），不能外推真实 int4 部署。`feats/` 不入库。

---

### 一句话串起来

另一台机先摸清「怎么采表示」；第 1 版判错了公式；第 2 版改对公式但样本不够；第 3 版严格重跑，才是现在跟导师讲的那套。

---

## 3. 另一台机的方法（`llm_transfer/`）

### 做了什么

把官方 `diffip.llm.LLMRepresentationCollector` 接到 LLM 上：对同一条 prompt，在 **输入 embedding 上加配对种子的高斯噪声**，hook 抓 decoder 层 hidden，attention-mask 加权 mean pool → `[L, N, D]`。再用官方 `DiffIPMetric.compare()` 打分（PCA-20 → 逐步 Procrustes → 可选 DP）。

### 依据

- DiffIP 要求 suspect / victim **同一组 seed、顺序一致**（官方 README；论文多样本估计）。
- 探索目标：找到可用的 `perturbation_scale`、验证真实 SFT 对能否排序、做层数/样本数消融与 ROC。

### 工作点与主要数字（均见 `REPORT.md`，勿口头改数）

| 项 | 值 |
|---|---|
| 工作点 | `perturbation_scale=2e-3`，`seed=123`，L=4，N=8 |
| SmolLM2-135M 自比 | 1.000 |
| SmolLM2-135M ↔ Instruct（真实 SFT） | **0.505** |
| SmolLM2-135M ↔ Qwen2.5-0.5B（跨家族） | **0.323** |
| 层/样本消融 | 该配置下裕度天花板约 **0.18**；N=16 时正例塌到约 0.04 |
| 混合采样（hybrid） | **否定**：异质 prompt 混进同一样本轴会毁配对假设 |
| ROC（2 正对 × 4 负对，逐 prompt） | **AUC 0.678** |

脚本：`sweep_perturbation.py`、`sweep_modes.py`、`validate_real_pair.py`、`ablate_layers_samples.py`、`ablate_hybrid.py`、`per_prompt_eval.py`、`roc_calibration.py`、`pilot_local_models.py`、`llm.py`。

**贡献**：确立了「配对扰动 + scale=2e-3」这条采样腿，并诚实写出 hybrid / 大 N 在 *DiffIPMetric 工作点* 下的塌陷。  
**局限**：主判定走的是封装好的 `DiffIPMetric`（含 PCA-20 与 DP）；样本量 N=8、负例设计与后文严格重跑不同；早期文档里还有用 margin 叙事的痕迹。

---

## 4. 本 bot 的方法（忠实 DiffIP：ALS + 平均 L2 + 大数定律）

### 依据（DiffIP 原文要点）

- 每个输入单独采 N 个成对种子样本，估计表示行为（大数定律）。
- 用 ALS 拟合正交 Q、对角 S、平移 h，把 suspect 表示拉回 victim（QSX−h）。
- **主指标是逐层平均 L2 距离**，相似度 `exp(−Σd/L)`。  
- **不用 margin 作判定**（margin 是自造的「正最低 − 负最高」，不是 DiffIP 公式）。
- DP 只在时间轴长度不同时才需要；LLM 等长层默认关 DP，剪枝可开对照。

### 三代目录分别干了什么

1. **最早版本 `llm_migration/`**（已否）  
   多 prompt、last-token / 混用 DiffIPMetric；主表用 **margin**；还有蒸馏、旋转攻击等旁支。被批评偏离 DiffIP，**保留仅作历史**。

2. **修正协议 `llm_migration_v2/`**  
   改回：同一 prompt + 配对扰动 N=8 + `fit_step_alignment` 的 mean L2；DP 关；范围仅微调 / 置换 / 缩放 / 剪枝。  
   PCA 维数扫描选出 **8 维**（勿盲用 20）。补了 5 个无关模型负例面板。  
   仍不够：只有 1 个 prompt，N=8 谈不上大数定律。

3. **严格重跑 `llm_migration_v3/`**（当前主结论）  
   - 32 个 prompt，3 个 seed set（123 / 1123 / 2123），N = 8…128  
   - victim = Qwen2.5-0.5B；衍生：Instruct 微调、权重/表示置换、各向异性缩放、剪枝；5 个无关负例  
   - 方案 (a) 配对 embedding 扰动为主；(b) 解码随机续写作对照  
   - 剪枝开/关 DP；`reef_baseline/` 同模型对上跑 REEF(CKA)

### 严格重跑关键数字（`tables_pert.md`，dims=8，跨 prompt 均值）

| | N=8 | N=20 | N=128 |
|---|---|---|---|
| 微调 / 置换 | 0.896 | 0.792 | **0.760** |
| 缩放 | ≈0.89 | ≈0.79 | **0.754** |
| 剪枝（DP 关） | 0.848 | 0.690 | **0.645** |
| 5 个负例 | 0.56–0.58 | ≈0.32 | **≈0.236** |

- N≥64 时跨 seed-set 标准差 ≤ 0.005。  
- N=20、dims=8：衍生模型在 **32/32** 个 prompt 上高于该 prompt 的全部 5 个负例。  
- **解码对照**：分数接近 0（微调约 0.02），配对失效 → 否定方案 (b)。  
- **REEF 官方设置**（TruthfulQA 末 token CKA）：衍生最低 − 负例最高 = **−0.036**，在这批 0.5B 上分不开剪枝与同族负例。  
- **同一份扰动云上算 CKA**：间隔约 +0.49，与 DiffIP 一样可分 → 区分力主要来自「成对扰动采样」，不能全归功于 ALS。

---

## 5. 两者的区别与联系

### 共同点

- 都把 LLM 表示整理成 `[L, N, D]`，层轴对应 DiffIP 的时间步。
- 都强调 **配对种子** 的 embedding 扰动（工作尺度 2e-3）。
- 都承认 hybrid / 非配对会毁协议；都不改上游 `diffip/` 包。

### 不同点

| 维度 | 另一台机 `llm_transfer/` | 本 bot 严格重跑 `llm_migration_v3/` |
|---|---|---|
| 判定公式 | `DiffIPMetric.compare()`（PCA-20 + Procrustes + 可选 DP） | 直接调用 ALS（QSX−h），**mean L2 → exp(−mean L2)**；主表不用 margin |
| 样本量 | N=8，单 seed 工作点 | N 扫到 128，3 个 seed set，显式检验大数定律 |
| Prompt | 少量 / 按脚本 | 固定 32 条，逐 prompt 打分再平均 |
| 负例 | 跨家族单点 + ROC 小集合 | 5 个无关模型面板，逐 prompt 可分计数 |
| 解码对照 | 无 | 有（方案 b，已否定） |
| REEF 对照 | 无（同设定） | `reef_baseline/` 同模型对 |
| PCA 维数 | 默认 20 | 扫描后主报 **8**（20 作附录） |
| 历史包袱 | 探索期有「天花板 ~0.18」、margin 叙事 | 最早版本用过 margin，**已明确否掉** |

### 怎么理解两条线的关系

1. **联系**：另一台机先把「配对扰动采集器」跑通，并定下 scale=2e-3；严格重跑沿用同一采集思想，但把**判定**拉回 DiffIP 原文的平均 L2，并把 N / prompt / 负例做大。  
2. **区别**：不是「旧版 vs 新版调参」，而是 **「封装 DiffIPMetric 的探索」vs「按论文公式的严格协议」**。最早 `llm_migration/` 又是第三条——用 margin 当 verdict，已被否，不要和另外两条混称。  
3. **数字不可直接横比**：`llm_transfer` 的 0.505（SmolLM2 SFT）与 v3 的 0.760（Qwen Instruct，N=128）victim、N、维数、打分路径都不同。

---

## 6. 怎么复现严格重跑

在仓库根目录，需已装 `torch` / `transformers` / 官方 DiffIP 依赖，并有对应模型权重（脚本默认读本地缓存；路径见 `common.py`）。

```bash
cd llm_migration_v3

# 1) 配对扰动采集 → feats/
python collect_perturb.py

# 2) ALS 打分（pert）与汇总表
python score.py pert
python aggregate.py          # → summary_pert.json, tables_pert.md

# 3) 剪枝 DP 开（可选）
python score_dp.py
python aggregate_dp.py       # → tables_dp.md

# 4) 解码对照（方案 b）
python collect_decode.py
python score.py dec
# 汇总见 tables_dec.md / summary_dec.json

# 5) REEF 基线（可选）
cd reef_baseline
python collect.py && python score_reef.py && python score_pert_cka.py && python make_tables.py
```

一键脚本：`run_all.sh`（扰动+解码）、`run_rest.sh`（续跑）。  
等价性检查：`test_equivalence.py`（批量实现 vs 官方 collector）。

日志、`feats/`、`acts/`、`__pycache__/` **不要提交**。

---

## 7. 建议阅读顺序

1. **本导读**（先看第 2 节「版本演进」，再看目录地图与对照表）  
2. [`llm_transfer/REPORT.md`](../llm_transfer/REPORT.md)（另一台机探索与数字）  
3. [`llm_migration_v3/实验报告.md`](../llm_migration_v3/实验报告.md)（严格重跑全文）  
4. [`llm_migration_v3/reef_baseline/说明.md`](../llm_migration_v3/reef_baseline/说明.md)（与 REEF/CKA）  
5. 需要协议细节时再看 [`llm_migration_v2/PROTOCOL.md`](../llm_migration_v2/PROTOCOL.md) 与 PCA 扫描  
6. [`llm_migration_v3/fingerinv_attacks/FingerInv攻击补测.md`](../llm_migration_v3/fingerinv_attacks/FingerInv攻击补测.md)（FingerInv 三类攻击补测）
7. [`llm_migration/`](../llm_migration/) 仅在需要「最早被否的设计」时翻

一句话收束：**配对扰动采样 + 足够大的 N，能分开衍生模型与无关模型；ALS 相对「同一扰动云上的 CKA」是否更强，尚未被证明——严格重跑已把这一点写进局限。**

## 8. 文献协议实验（REEF / HuRef 式评测）

详见 [`llm_migration_v3/lit_protocol/文献协议实验.md`](../llm_migration_v3/lit_protocol/文献协议实验.md)。这一批做了五件事：
- 按 REEF（ICLR'25）和 HuRef（NeurIPS'24）的做法，拿 9 个 HF 上真实的 Qwen2.5-0.5B 后代当正例；
- 加入 2 个同厂商难负例（Qwen2-0.5B-Instruct、Qwen3-0.6B-Base）；
- 每行给出置换检验 p 值；
- 做了基座归属实验，但因为相似度的底噪随参考模型变化，这个实验作废；
- 蒸馏换成输出信号重测。

middle2 下的结果：
- SFT、权重编辑、合并类的 7/7 个后代全部检出（p=5e-05）；
- 难负例没有出现假阳性；
- Qwen2.5-Coder 系列（继续预训练）和 R1-DRAFT 失败，跌到负例水平。

蒸馏方面，"教师回答 NLL"信号对 control 学生的 AUC 为 0.89，扰动云分布信号仍只有 0.52。
