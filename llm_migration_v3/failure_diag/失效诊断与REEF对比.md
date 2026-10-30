# 失效诊断与 REEF 对比（failure_diag）

日期：2026-10-30（Asia/Shanghai）。所有数字来自本目录已提交的 json / log 或下文"既往结果"里链接的已提交文件。

| 文件 | 作用 |
|---|---|
| `static_diag.py` → `results_static.json` | 分词器、嵌入矩阵、逐层权重相对变化（从 safetensors 逐张量读取，不做前向） |
| `dyn_collect.py` → `feats/*.npz`（gitignore） | 与 `lit_protocol/collect.py` 相同的扰动协议，另加种子集 0 的全层特征；包含换分词器 / 换嵌入两种变体 |
| `dyn_score.py` → `results_dynamic.json`, `dyn_score.log` | DiffIP 打分（all4 / middle2）和逐层 sim 曲线 |
| `reef_run.py` → `results_reef.json`, `reef.log` | REEF 官方设定 + 扰动云上的 CKA（公式与 `reef_baseline` 相同） |

## 1. 三个"贴地"模型的诊断

背景：在 [`lit_protocol`](../lit_protocol/文献协议实验.md) 中，Coder-0.5B、Coder-0.5B-Instruct 和 R1-DRAFT 的 middle2 sim 分别是 .154 / .148 / .144，和负例下限（≈0.143）几乎一样。

### 1.1 静态对比（vs Qwen2.5-0.5B）

| 模型 | 词表一致比例 | 32 个 prompt 的 token id 完全相同 | 嵌入逐行 cos（全词表 / prompt 内 token） | 逐层权重相对变化 ‖ΔW‖/‖W‖（范围；中位层约值） |
|---|---:|---:|---|---|
| Coder-0.5B | 1.000 | 32/32 | 0.504 / 0.726 | 0.24–0.64，约 0.55 |
| Coder-0.5B-Inst | 1.000 | 32/32 | 0.341 / 0.567 | 0.29–0.85，约 0.68 |
| **R1-DRAFT** | 0.99995 | **0/32** | 0.997 / 0.995 | **0.011–0.052，约 0.04** |
| 对照 Instruct | 1.000 | 32/32 | 0.993 / 0.998 | 0.015–0.054 |
| 对照 Vikhr | 1.000 | 32/32 | 0.993 / 0.998 | 0.05–0.21 |
| 负例 Qwen2-0.5B | 0.9999 | 32/32 | −0.011 / −0.004 | 1.00–1.51 |

R1-DRAFT 的分词器会在每条输入前加 BOS `<｜begin▁of▁sentence｜>`（id 151646；在 victim 词表里，这个 id 是 `<|object_ref_start|>`）。去掉这个 BOS 后，32 条 token 序列与 victim 完全一致（32/32）。

### 1.2 换分词器 / 换嵌入实验（DiffIP，seed set 0，N=128，d=8；负例最大值来自 base5）

| 变体 | all4 sim / gap / AUC | middle2 sim / gap / AUC |
|---|---|---|
| r1_draft（原样） | .234 / −.001 / .499 | .144 / −.000 / .496 |
| **r1_draft + victim 分词器（不加 BOS）** | **.633 / +.397 / .906** | **.645 / +.501 / .946** |
| victim + R1 分词器（加 BOS，对照） | .233 / −.002 / .493 | .143 / −.001 / .495 |
| coder_base（原样） | .247 / +.012 / .532 | .154 / +.010 / .549 |
| coder_base + victim 嵌入 | .238 / +.003 / .509 | .147 / +.003 / .521 |
| coder_inst（原样） | .238 / +.003 / .509 | .148 / +.003 / .525 |
| coder_inst + victim 嵌入 | .237 / +.002 / .509 | .147 / +.003 / .518 |
| instruct（正例对照） | .758 / +.522 / .961 | .766 / +.622 / .989 |
| qwen2_inst（负例对照） | .234 / −.001 / .498 | .143 / −.001 / .496 |

原样运行的结果与 lit_protocol 一致（.154 / .148 / .144），说明这次重新采集可以复现之前的数字。

### 1.3 逐层 sim 曲线（单层，24 层，seed set 0；完整数据见 `results_dynamic.json` 的 `layer_curve`）

- 第 0–1 层：所有模型都是 ≈0.98–1.0，**负例 qwen2_inst 也是 0.991 / 0.983**。这两层没有区分力。第 21 层也有类似情况，所有模型约 0.50。
- 第 2–20 层：instruct 0.67–0.78；r1+victim 分词器 0.51–0.66；Coder 两个模型 0.11–0.16，**与 qwen2_inst（0.10–0.15）重合**。Coder 在哪一层都没有残留信号。

### 1.4 假设判定

| 假设 | 判定 | 证据 |
|---|---|---|
| H1 R1-DRAFT 贴地是因为输入 token 不一致（分词器多了 BOS） | **支持** | 只换分词器：middle2 从 .144 升到 .645，gap +.501，AUC .946；反过来给 victim 自己加 BOS 也掉到 .143 |
| H2 R1-DRAFT 的权重被大幅改动 | **否定** | 逐层 ΔW 约 4%，和官方 Instruct 同一量级；嵌入 cos .997 |
| H3 Coder 失效是因为嵌入层或分词器变了 | **否定** | 分词结果 32/32 相同；换入 victim 嵌入后仍是 .147 |
| H4 Coder 失效是因为大规模继续预训练造成了整体权重漂移 | **支持（相关性证据）** | 每一层 ΔW 都在 0.24–0.85，比 Instruct 大 10 倍以上，介于 Vikhr（≤0.21，可检出）和独立预训练的 Qwen2（≥1.0）之间；sim 降到负例水平 |
| H5 只有个别层漂移，换一组层就能救回来 | **否定** | 第 2–20 层全部在负例水平；第 0、1、21 层所有模型（包括负例）都高，没有区分力 |

结论：
- **R1-DRAFT 的失败是协议层面的问题**。DiffIP 要求 suspect 和 victim 的输入 token 完全一致，多一个 BOS 就会让整条序列的位置和注意力上下文都变掉。只要统一用 victim 的分词器，它就能被检出。
- **Coder 的失败是真实的漏检**。约 5.5T token 的继续预训练（见 `lit_models.py` 的注释）已经把成对扰动的几何结构改到了和独立模型一样，DiffIP 对这种程度的"重度继续预训练"不鲁棒。

## 2. REEF 对比（`reef_run.py`）

- **官方设定**：TruthfulQA200、最后一个 token、相对深度 18/31 处的 linear CKA。AUC 用 10 块×20 条，和 base5 负例的 chunk 比较。
- **扰动云 CKA**：用 lit_protocol 的扰动特征，N=128，4 层平均。AUC 用逐 prompt 值。
- gap 一律等于"该模型 − base5 负例最大值"：官方设定的负例最大值是 .910（Qwen2-0.5B），扰动云是 .297。
- DiffIP 一栏是 lit_protocol 的 middle2 结果；其中 r1 一行额外附上了换 victim 分词器后的结果。

| 模型 | 标签 | REEF 官方 linear / gap / AUC | REEF 官方 RBF | 扰动云 CKA / gap / AUC | DiffIP middle2 sim / gap / AUC |
|---|---|---|---:|---|---|
| coder_base | 正 | .938 / +.028 / .998 | .953 | .306 / +.009 / .751 | .154 / +.010 / .549 |
| coder_inst | 正 | .895 / **−.015** / .642 | .917 | .266 / −.031 / .501 | .148 / +.003 / .525 |
| r1_draft | 正 | .961 / +.050 / 1.000 | .970 | .253 / −.044 / .423 | .144 / −.000 / .496（换分词器后 .645 / +.501 / .946） |
| vikhr_sft | 正 | .945 / +.035 / 1.000 | .959 | .708 / +.411 / 1.000 | .426 / +.282 / .867 |
| ita_sft | 正 | .968 / +.057 / 1.000 | .975 | .803 / +.506 / 1.000 | .594 / +.450 / .951 |
| qwen2_inst | 负 | .908 / −.003 / .850 | .930 | .301 / +.004 / .722 | .143 / — / .497 |
| qwen3_base | 负 | .904 / −.006 / .792 | .929 | .275 / −.022 / .565 | .143 / — / .490 |

要点：
1. **REEF 官方设定能把 R1-DRAFT 和 Coder-base 排在负例之上**（.961 / .938 vs 负例最大 .910）。原因是它只取最后一个 token，而且对输入 token 不严格对齐，所以 BOS 不影响它。但它的余量很小（+.03 到 +.06）：Coder-Inst（.895）低于难负例 Qwen2-Inst（.908），这时判错。
2. **扰动云 CKA 和 DiffIP 的表现一致**：Vikhr 和 ITA 都检出，余量大；Coder 和 R1 都贴地。R1 贴地也是因为特征带了 BOS。这再次说明，可分性主要来自"成对扰动 + 输入严格一致"这一数据设计，而不是 ALS 还是 CKA 这一度量选择。
3. 取舍：REEF 官方设定覆盖面广但余量薄，容易被难负例超过；DiffIP 和扰动云 CKA 余量厚，但要求输入 token 完全一致，对重度继续预训练不鲁棒。

## 3. 既往结果汇总（只引用已提交文件）

| 实验 | 关键数字 | 出处 |
|---|---|---|
| v3 主表 all4 vs middle2 | gap .411 → .605；AUC .953 → .989；prune .646 → .749 | [`improve/主表_middle2.md`](../improve/主表_middle2.md)、[`tables_pert_middle2.md`](../tables_pert_middle2.md) |
| REEF 基线（v3 面板） | 官方 gap **−0.036**（不可分）；扰动云 CKA gap **+0.488** | [`reef_baseline/说明.md`](../reef_baseline/说明.md) |
| improve：QSh vs 余弦 | 1 向量/prompt 时余弦 AUC 1.0、QSh ≈.52；含置换时余弦 .600，扰动云 QSh 1.000；middle2 prune gap .394 → .601 | [`improve/提升报告.md`](../improve/提升报告.md) |
| 蒸馏代理 | DiffIP 区分 distill vs teacher：AUC .504（≈0.50，负结果） | [`improve/蒸馏鲁棒性.md`](../improve/蒸馏鲁棒性.md) |
| FingerInv 攻击（359d8f6） | middle2：fp16 .763 / bf16 .580 / int8dyn .195 / w8sim .767 / w4sim .309；层剪枝 5–30% .763–.740，50% .231；幅值剪枝 30% .366、50% .202 | [`fingerinv_attacks/FingerInv攻击补测.md`](../fingerinv_attacks/FingerInv攻击补测.md) |
| 文献协议（4b27b75） | 7 个真实后代可检出（NuExtract .984 … Vikhr .426，p=5e-5）；Coder ×2、R1 贴地；合并 AUC .817；溯源无效；蒸馏输出信号 S2 teacher-NLL AUC .891 | [`lit_protocol/文献协议实验.md`](../lit_protocol/文献协议实验.md) |
| **本次失效诊断** | R1 改用 victim 分词器后 middle2 .645 / +.501 / .946；Coder 换嵌入后仍为 .147 | 本文 §1 |
| **本次 REEF 对比** | 官方 R1 .961、Coder .938、Coder-Inst .895 < Qwen2-Inst .908 | 本文 §2 |

## 4. 偏差与限制

- 第 0、1 层（以及第 21 层）对所有模型都高，所以单层曲线只能看第 2–20 层。
- H4 只有相关性证据：没有训练"不同继续预训练量"的中间模型（CPU 环境做不到）。
- REEF 的 AUC 用 10 块×20 条（与 `reef_baseline` 相同），样本量小。
- 没有对其余 lit 模型做"统一分词器"的重测：它们的 token id 本来就和 victim 一致（Vikhr 已核对，32/32）。
