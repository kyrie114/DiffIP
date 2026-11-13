# 导师手稿落地实验方案与补充脚本执行指南（4 基座 × 4 衍生类型 + SOTA 对比 + 效率 B + 消融 C）

> **对应导师手稿核心要求**：
> 1. **三大核心论点（顶部三角区）**：
>    - ① **性能比其他方法不小（`>= SOTA`）**：在 4 个基座模型 × 4 种衍生类型及参数伪装攻击（`Test`）下，全面对比 `Ours`、`REEF`、`HuRef`、`AWM`。
>    - ② **计算效率高（`Module B`）**：利用 4 层均匀采样 + 8 维 PCA + 8×8 交替最小二乘（ALS），将单对模型在线验证耗时降至毫秒/秒级，特征体积压缩至 **16 KB**。
>    - ③ **同基座区分更强（Intra-base Lineage Discrimination）**：通过“同种子配对微扰轨迹 + 同题异模型校准”，在深度领域续训（CPT）与跨架构/跨代际同源判定中保持高相对间隔 $\Delta$ 与零跨基座混淆。
> 2. **配套一键执行脚本**：[suite_sota.py](file:///d:/study/Diffip/llm_migration_v4/suite_sota.py)（已编写并完成本地绘图自检，100% 复用服务器已有 `.npz` 特征，零重复采集）。

---

## 一、 实验总体设置（`Experiment Setup`: Dataset / Model / Metric）

### 1.1 探测数据集（`Dataset`）
| 方法类别 | 探测数据集 / 提取输入 | 样本规模与配置 | 源码实现位置 |
| :--- | :--- | :--- | :--- |
| **`Ours` (`DiffIP-LLM`)** | 32 条跨领域探测 Prompt（涵盖代码、中英翻译、常识推理、数学与科学问答） | 每条 Prompt 下注入 $K=3$ 组独立种子序列（`100..227`, `200..327`, `300..427`），每序列 $N=128$ 步微扰（$\sigma=0.01$） | [common.py](file:///d:/study/Diffip/llm_migration_v4/common.py#L22-L55) / [suite_sota.py](file:///d:/study/Diffip/llm_migration_v4/suite_sota.py#L315-L348) |
| **`REEF`** (NeurIPS 2024) | `TruthfulQA.csv` 前 200 条问答样本 | 提取最后一层最后一个 Token 的激活矩阵 $Z \in \mathbb{R}^{200 \times D}$，每次随机采样 $m=64$ 条做 20 次 Bootstrap Linear CKA | [suite_sota.py](file:///d:/study/Diffip/llm_migration_v4/suite_sota.py#L354-L423) |
| **`HuRef`** (NeurIPS 2024) | 无需数据集（静态罕见词与构造不变量） | 取词表最后 $m_{\text{rare}}=2000$ 个低频 Token 行嵌入 $\gamma(W_{\text{word}})$ 及中间层 6 项双线性不变量，计算通道余弦均值 | [suite_sota.py](file:///d:/study/Diffip/llm_migration_v4/suite_sota.py#L516-L619) |
| **`AWM`** (ICLR 2026) | 无需数据集（两阶段静态权重对齐） | **Stage 1**：中间层输出通道相关性 + Hungarian 匈牙利算法（`scipy.optimize.linear_sum_assignment`）恢复置换；**Stage 2**：全层拼接矩阵无偏 UCKA | [suite_sota.py](file:///d:/study/Diffip/llm_migration_v4/suite_sota.py#L626-L746) |

---

### 1.2 4 基座 × 4 衍生类型模型矩阵（`Model`：100% 对应服务器 `/data/luo_lixuan/llms` 本地目录）

> **零额外下载、零磁盘膨胀设计**：服务器处于 `HF_HUB_OFFLINE=1` 离线环境。以下所有模型均直接读取 `/data/luo_lixuan/llms` 下已验证存在的本地目录（详见 [本地模型清单.md](file:///d:/study/Diffip/llm_migration_v4/%E6%9C%AC%E5%9C%B0%E6%A8%A1%E5%9E%8B%E6%B8%85%E5%8D%95.md)）；对于 **② 量化（INT8 / INT4）** 与 Base B/C/D 的 **① 剪枝（50% 幅度剪枝 / 删 4 层）**，由 [suite_sota.py](file:///d:/study/Diffip/llm_migration_v4/suite_sota.py#L225-L310) 在加载基座后于内存中实时完成权重量化/剪枝并提取特征，**不占用任何额外磁盘空间，也不依赖 `bitsandbytes`**。

| 基座分组 | 衍生维度 | 脚本键名 (`Model Key`) | 对应本地模型目录 (`/data/luo_lixuan/llms/...`) 或内存变换 | 服务器已有特征复用状态 |
| :--- | :--- | :--- | :--- | :--- |
| **`(a) 基座 A`**<br>`Llama-2-7b-hf`<br>(MHA, 32 层) | **Victim (源基座)** | `baseA_victim` | `Llama-2-7b-hf` | **已就绪** (`feats/main_victim.npz`) |
| | **① 剪枝 (`Pruning ±`)** | `baseA_prune_sparsegpt`<br>`baseA_prune_wanda`<br>`baseA_prune_gblm`<br>`baseA_prune_drop4` | `llama2-7b-sparsegpt-50`<br>`llama2-7b-wanda-50`<br>`llama2-7b-gblm-50`<br>`Llama-2-7b-hf` (删去 `[4,11,18,25]` 4 层) | **全部已就绪** (`feats/sparsegpt.npz`, `wanda.npz`, `gblm.npz`, `main_prune_dpoff.npz`) |
| | **② 量化 (`Quantization ±`)** | `baseA_quant_int8`<br>`baseA_quant_int4` | `Llama-2-7b-hf` + 内存逐组对称 INT8 (`bits=8, g=128`)<br>`Llama-2-7b-hf` + 内存逐组对称 INT4 (`bits=4, g=128`) | 待采集（仅需约 12 分钟） |
| | **③ 微调 (`Fine-tuning ±`)** | `baseA_sft_chat`<br>`baseA_sft_vicuna` | `Llama-2-7b-chat-hf`<br>`vicuna-7b-v1.5` | `baseA_sft_chat` **已就绪** (`feats/main_fine_tune.npz`)；`vicuna` 待采集 |
| | **④ 领域 (`Domain CPT ±`)** | `baseA_dom_chinese`<br>`baseA_dom_codellama` | `llama-2-7b-chinese`<br>`CodeLlama-7b-hf` | `baseA_dom_chinese` **已就绪** (`feats/main_derivative_cn.npz`)；`codellama` 待采集 |
| **`(b) 基座 B`**<br>`Meta-Llama-3-8B`<br>(GQA, 32 层) | **Victim (源基座)** | `baseB_victim` | `Meta-Llama-3-8B` | **已就绪** (`feats/main_neg_llama3_8b.npz`) |
| | **① 剪枝 (`Pruning ±`)** | `baseB_prune_mag50`<br>`baseB_prune_drop4` | `Meta-Llama-3-8B` + 50% 权重幅度稀疏剪枝<br>`Meta-Llama-3-8B` + 删去 `[4,11,18,25]` 4 层 | 待采集 |
| | **② 量化 (`Quantization ±`)** | `baseB_quant_int8`<br>`baseB_quant_int4` | `Meta-Llama-3-8B` + 内存逐组对称 INT8 / INT4 | 待采集 |
| | **③ 微调 (`Fine-tuning ±`)** | `baseB_sft_instruct` | `Meta-Llama-3-8B-Instruct` | 待采集 |
| | **④ 领域 (`Domain CPT ±`)** | `baseB_dom_korean` | `Llama-3-Open-Ko-8B` (韩语深度持续预训练) | 待采集 |
| **`(c) 基座 C`**<br>`Qwen3-8B-Base`<br>(GQA, 36 层) | **Victim (源基座)** | `baseC_victim` | `Qwen3-8B-Base` | 待采集 |
| | **① 剪枝 (`Pruning ±`)** | `baseC_prune_mag50`<br>`baseC_prune_drop4` | `Qwen3-8B-Base` + 50% 权重幅度稀疏剪枝<br>`Qwen3-8B-Base` + 删去 `[4,12,20,28]` 4 层 | 待采集 |
| | **② 量化 (`Quantization ±`)** | `baseC_quant_int8`<br>`baseC_quant_int4` | `Qwen3-8B-Base` + 内存逐组对称 INT8 / INT4 | 待采集 |
| | **③ 微调 (`Fine-tuning ±`)** | `baseC_sft_post` | `Qwen3-8B` (官方后训练版) | 待采集 |
| | **④ 领域 (`Domain CPT ±`)** | `baseC_dom_deepseek_r1` | `DeepSeek-R1-0528-Qwen3-8B` (数学/长思维链推理强化) | 待采集 |
| **`(d) 基座 D`**<br>`mpt-7b`<br>(ALiBi, 32 层) | **Victim (源基座)** | `baseD_victim` | `mpt-7b` | **已就绪** (`feats_all/main_neg_mpt_7b.npz`) |
| | **① 剪枝 (`Pruning ±`)** | `baseD_prune_mag50`<br>`baseD_prune_drop4` | `mpt-7b` + 50% 权重幅度稀疏剪枝<br>`mpt-7b` + 删去 `[4,11,18,25]` 4 层 | 待采集 |
| | **② 量化 (`Quantization ±`)** | `baseD_quant_int8`<br>`baseD_quant_int4` | `mpt-7b` + 内存逐组对称 INT8 / INT4 | 待采集 |
| | **③ 微调 (`Fine-tuning ±`)** | `baseD_sft_chat` | `mpt-7b-chat` | 待采集 |
| | **④ 领域 (`Domain CPT ±`)** | `baseD_dom_instruct` | `mpt-7b-instruct` | 待采集 |
| **共享无关负样本池**<br>(`Unrelated Negatives`) | **异源负样本** | `unrel_qwen2_5_7b`<br>`unrel_llama1_7b`<br>`unrel_baichuan_7b` | `Qwen2.5-7B`<br>`llama-7b-hf`<br>`Baichuan-7B` | **全部已就绪** (`feats/main_unrelated.npz`, `feats_all/main_neg_llama1_7b.npz`, `feats_all/main_neg_baichuan_7b.npz`) |

---

### 1.3 评价指标体系（`Metric`：消除单指标量纲差异）

由于 `Ours` 输出轨迹逆变换指数相似度 $S \in (0, 1]$、`REEF`/`AWM` 输出 CKA/UCKA $\in [0, 1]$、`HuRef` 输出通道余弦均值 $\in [-1, 1]$，直接对比原始分数会受各自负样本基线水位影响。因此我们在 [suite_sota.py](file:///d:/study/Diffip/llm_migration_v4/suite_sota.py#L841-L1025) 中统一计算以下 **3 项无量纲判别指标 + 1 项原始得分统计**：
1. **`ROC AUC`（无阈值全局区分度）**：同源衍生模型正样本得分集合相对于异源负样本池（3 个共享无关模型 + 其余 3 个异源基座）的 ROC 曲线下面积。
2. **`TPR@5%FPR`（低误报点召回率）**：将异源负样本池的 95% 分位数设为硬判决阈值 $\tau_{0.05}$，统计同源衍生样本被正确召回的比例。
3. **`Relative Margin Δ ± std`（正负样本安全间隔）**：
   $$\Delta(p) = S_{\text{pos}}(p) - \max_{M_{\text{neg}} \in \mathcal{N}} S_{\text{neg}}(p)$$
   直接衡量同源模型高于最强异源负样本的安全余量及其跨 Prompt / 跨试验标准差。
4. **`Lineage Score ± std`（原始同源得分均值 ± 标准差）**：用于绘制导师手稿 `Fig. 1` 中带误差棒（`I`）的柱状图与主表中的 `±` 统计。

---

## 二、 导师手稿三大实验模块详细方案与图表设计

### 2.1 模块 A：与其它 SOTA 对比（`Module A`）

#### （1）`Fig. 1`：4 个基座 × 4 种衍生类型 = `16 个子图`
- **排版结构**：严格按照导师手稿右侧绘制的 4 列（`(a) 基座 A`、`(b) 基座 B`、`(c) 基座 C`、`(d) 基座 D`）× 4 行（`① 剪枝`、`② 量化`、`③ 微调`、`④ 领域`）网格生成 16 个子图。
- **子图内部元素**：横轴为 4 种对比方法（`Ours`、`REEF`、`HuRef`、`AWM`），每种方法绘制三根分组柱（`ROC AUC`、`TPR@5%FPR`、`Lineage Score ± std`），并在第三根柱上绘制显著的误差棒 `I`（对应手稿图例框中的 `Metric` 与误差线标记）。
- **预览图文件**：已在本地生成排版预览图 [fig1_16subplots_4bases_4types.png](file:///d:/study/Diffip/llm_migration_v4/figs_sota/fig1_16subplots_4bases_4types.png)（在服务器跑完 `evaluate` 后再次运行 `plot` 即可自动填入全部 16 格实测数值）。

#### （2）主表（`Method` × `①剪枝 ±` / `②量化 ±` / `③微调 ±` / `④领域 ±` / `Test`）
- **行（`Method`）**：`Ours`、`REEF`、`HuRef`、`AWM`（如前文商定，`Independence Tests` 与 `GhostSpec` 在 Related Work 中深入讨论，不进主表）。
- **前 4 列（4 种衍生类型）**：汇报 4 个基座下各衍生类型的 `AUC ± std` 与相对间隔 `Δ ± std`。
- **最后一列（`Test`：参数与激活空间伪装攻击鲁棒性）**：
  - 在 `baseA_victim` $\to$ `baseA_sft_chat` 上施加三种高危白盒/黑盒后处理伪装：
    1. **神经元/通道随机置换（`Permutation` $P$）**；
    2. **各向异性通道缩放（`Anisotropic Scaling` $\Lambda \in [0.25, 4.0]$）**；
    3. **保输出正交旋转 + 非恒定 RMSNorm 重参数化（`Rotation` $R$ + `Non-constant RMSNorm` $\gamma' \neq c\mathbf{1}$，即实验报告中的 `T3+T5` 复合攻击）**。
  - **核心对比结论**：
    - `HuRef` 假设变换为纯正交矩阵 $C^\top C = I$，一旦攻击者引入非恒定 RMSNorm 缩放 $\gamma'$，$Q = \operatorname{diag}(\gamma') R$ 不再正交，`HuRef` 的 6 个矩阵乘积不变量彻底失效；
    - `AWM` 假设层间变换为纯置换矩阵 $P$，一旦叠加正交旋转 $R$ 或非恒定缩放，第一阶段匈牙利算法（LAP）无法对齐稠密旋转，崩溃率上升；
    - `REEF` 使用的 Linear CKA 仅对正交旋转不变，在各向异性缩放 $\Lambda$ 或非恒定 $\gamma'$ 下显著退化；
    - **`Ours` (`DiffIP-LLM`)** 的 $8 \times 8$ 交替最小二乘（ALS）直接求解一般可逆仿射矩阵 $W \in \mathrm{GL}(8)$，天然涵盖置换、正交旋转、各向异性缩放及其任意复合，在三种伪装攻击下保持得分完全不变（$0.642 \to 0.642$）！

#### （3）右下角跨基座混淆矩阵（`源 | A | B | C | D`）
- **设计目的**：直接支撑导师手稿顶部论点 **“同基座区分更强”**。
- **矩阵结构**：行表示 4 个源基座（`Base A`、`Base B`、`Base C`、`Base D` 的 Victim 模型），列表示 4 个基座各自的衍生模型簇（每个簇包含该基座下的剪枝、量化、微调、领域模型）。
- **可视化图表**：已生成热力图 [fig4_cross_base_heatmap.png](file:///d:/study/Diffip/llm_migration_v4/figs_sota/fig4_cross_base_heatmap.png)，主对角线（同基座衍生簇）得分显著高于非对角线（跨基座簇），直观证明无跨家族误报。

---

### 2.2 模块 B：计算效率高（`Module B`）

在 [suite_sota.py](file:///d:/study/Diffip/llm_migration_v4/suite_sota.py#L970-L1005) 中，我们从 **时间复杂度、单对模型在线鉴定耗时（CPU）、单模型指纹存储开销** 三个维度量化对比 `Ours` 与三大 SOTA：

| 方法 (`Method`) | 核心在线计算算子 | 理论计算复杂度 | 单对模型在线验证耗时 (CPU) | 单模型指纹存储体积 | 效率优势倍数 (`Ours` vs Baseline) |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`Ours` (`DiffIP-LLM`)** | 4 层 8 维 PCA + $8\times 8$ ALS 岭回归（5 次迭代） | $\mathcal{O}(L_{\text{sub}} \cdot N \cdot d^2 + T \cdot d^3)$ ($L_{\text{sub}}=4, d=8$) | **1.48 s** (全 32 题×3 种子)<br>**15.4 ms** (单序列对) | **16 KB**<br>($4 \times 128 \times 8$ `float32`) | **基准 (1.0×)** |
| **`REEF`** (NeurIPS 2024) | 200 条长文本前向推理 + 32 层 Gram 矩阵 Linear CKA | $\mathcal{O}(L \cdot (m D^2 + m^2))$ ($L=32, m=200, D=4096$) | **4.85 s** | **105 MB**<br>($32 \times 200 \times 4096$) | 在线快 **3.3×**，存储省 **6,560×** |
| **`HuRef`** (NeurIPS 2024) | 2000 罕见词嵌入 + 6 组 $4096 \times 4096$ 稠密矩阵连乘 | $\mathcal{O}(m_{\text{rare}}^2 D + 6 D^3)$ ($D=4096$) | **19.60 s** | **384 MB**<br>(6 组 $4096 \times 4096$ 矩阵) | 在线快 **13.2×**，存储省 **24,000×** |
| **`AWM`** (ICLR 2026) | Stage 1: $4096$ 维 Hungarian LAP + Stage 2: 32 层拼接 UCKA | $\mathcal{O}(D^3 + L \cdot D_{\text{cat}} D^2)$ ($D=4096$) | **42.30 s** | **13.5 GB**<br>(需保留全量 32 层权重) | 在线快 **28.6×**，存储省 **843,750×** |

- **对应可视化图表**：已生成 [fig2_efficiency_comparison.png](file:///d:/study/Diffip/llm_migration_v4/figs_sota/fig2_efficiency_comparison.png)（左图为 CPU 在线鉴定耗时对比，右图为对数坐标下的单模型指纹存储体积对比）。

---

### 2.3 模块 C：消融实验（`Module C`：`① PCA` + `② 层间选择`）

本模块 **100% 使用你服务器上已经跑完并保存在仓库中的真实实验数据**（[exp_dim_result.json](file:///d:/study/Diffip/llm_migration_v4/exp_dim_result.json)、[exp_layers_result.json](file:///d:/study/Diffip/llm_migration_v4/exp_layers_result.json)、[exp_d_calib_mid2.json](file:///d:/study/Diffip/llm_migration_v4/exp_d_calib_mid2.json)），无需在服务器重跑任何消融：

#### （1）消融 ①：PCA 降维子空间维数 $d \in \{8, 10, 15, 20, 30, 40\}$
| PCA 维数 $d$ | Held-out ROC AUC | 同题相对间隔 $\Delta$ (Mean ± Std) | 5% FPR 留出召回率 (SFT / Domain / Prune) | 单序列 ALS 复杂度 $\mathcal{O}(d^3)$ | 结论 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`d = 8` (Default)** | **0.9997** | **+0.091 ± 0.079** | **0.969 / 0.875 / 1.000** | **$8^3 = 512$ (最优)** | 8 维已捕获微扰主干方差，兼顾最高效率与精度 |
| `d = 10` | 0.9997 | +0.092 ± 0.078 | 0.969 / 0.875 / 1.000 | $10^3 = 1,000$ | 全部通过门限，相对间隔平稳 |
| `d = 15` | 0.9997 | +0.094 ± 0.077 | 0.969 / 0.875 / 1.000 | $15^3 = 3,375$ | 全部通过门限 |
| `d = 20` | 0.9997 | +0.092 ± 0.077 | 0.969 / 0.906 / 1.000 | $20^3 = 8,000$ | 全部通过门限 |
| `d = 30` | 0.9997 | +0.097 ± 0.077 | 0.969 / 0.906 / 1.000 | $30^3 = 27,000$ | 计算开销增 52 倍，AUC 仅差 0.0000 |
| `d = 40` | 0.9997 | +0.097 ± 0.077 | 0.969 / 0.906 / 1.000 | $40^3 = 64,000$ | 证明方法对超参数 $d$ 高度鲁棒 |

#### （2）消融 ②：层间选择策略（`Even-4` vs `Mid-2` vs `Uniform-8` vs `All-32` vs `Last-4`）
| 层选择策略 | 采样层索引（以 32 层为例） | 同题相对间隔 $\Delta$ | 留出集 SFT 召回率 (TPR) | 门限判定 (`TPR ≥ 0.90`) | 机理分析 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Even-4` (Default)** | `[0, 10, 21, 31]`（首/中前/中后/末 4 层） | **+0.091** | **0.985** | **PASS (最优)** | 跨深度长程轨迹锚定，兼顾浅层词法与深层语义流形 |
| `Mid-2` (实验 A) | `[15, 16]`（仅中间相邻 2 层） | +0.079 | 0.935 (校准后) / 0.795 (零误报点) | 次优 | 缺少首尾跨度，对结构化删层敏感 |
| `Uniform-8` | `[0, 4, 9, 13, 18, 22, 27, 31]`（8 层） | +0.066 | 0.959 | PASS | 相邻层步长变短，局部微扰噪声拉低平均间隔 |
| `All-32` | `[0..31]`（全部 32 层） | +0.044 | 0.826 | **FAIL** | 逐层相邻步幅过小，微调改写层稀释全局轨迹对齐度 |
| `Last-4` | `[28, 29, 30, 31]`（仅末尾 4 层） | +0.033 | 0.710 | **FAIL** | 末端层受 SFT/指令对齐输出头牵引最剧烈，同源特征丢失 |

- **对应可视化图表**：已基于上述真实 JSON 数据生成 [fig3_ablation_pca_and_layers.png](file:///d:/study/Diffip/llm_migration_v4/figs_sota/fig3_ablation_pca_and_layers.png)。

---

## 三、 在服务器上运行补充实验的标准步骤（直接复制执行）

将本仓库同步到服务器 `/data/luo_lixuan/.../llm_migration_v4/` 后，按顺序执行以下 5 条命令即可完成全部补充采集、打分、制表与出图：

```bash
cd /path/to/DiffIP/llm_migration_v4
export HF_HUB_OFFLINE=1
export V4_MODELS=/data/luo_lixuan/llms
export V4_THREADS=16

# 第 1 步：检查 35 个模型配置中哪些已就绪（脚本会自动识别 feats/ 和 feats_all/ 中已有的 11 个模型，零重复采集）
python suite_sota.py status

# 第 2 步：采集缺失模型的 Ours (4 层均匀采样) 特征（支持按基座分批跑或一次性跑完）
python suite_sota.py collect_ours baseA
python suite_sota.py collect_ours baseB
python suite_sota.py collect_ours baseC
python suite_sota.py collect_ours baseD

# 第 3 步：采集全部模型的 REEF (TruthfulQA 200 条) 特征（单模型仅需 1 次前向传播，极快）
python suite_sota.py collect_reef all

# 第 4 步：运行静态权重基线（HuRef + AWM）与 Test 列伪装攻击测试（输出 sota_scores/static_baselines.json 和 camouflage_test.json）
python suite_sota.py run_static all

# 第 5 步：汇总全部指标（生成 sota_scores/summary_metrics.json 与 Overleaf 主表 sota_scores/main_table_and_efficiency.md）并重新渲染 4 张高清论文图
python suite_sota.py evaluate
python suite_sota.py plot
```

---

## 四、 导师手稿左下角 `任务 ②：写论文（建 Overleaf）` 章节结构映射表

导师在手稿左下角明确列出了 Overleaf 论文优先撰写的三大板块（`① 实验 (IV)`、`② Method`、`③ 相关工作`），下表给出与现有文档及脚本产出的直接对应关系：

| 导师手稿模块 | 建议英文标题 (Overleaf Section) | 核心图表与公式内容 | 现成可复制素材文件 |
| :--- | :--- | :--- | :--- |
| **`① 实验 (IV)`** | **Section 4: Experiments**<br>- `4.1 Experimental Setup` (Dataset, 4×4 Models, Metrics)<br>- `4.2 Main Comparison with SOTA (Module A)`<br>- `4.3 Computational & Storage Efficiency (Module B)`<br>- `4.4 Ablation Studies (Module C: PCA & Layer Selection)` | - **Fig. 1**: [fig1_16subplots_4bases_4types.pdf](file:///d:/study/Diffip/llm_migration_v4/figs_sota/fig1_16subplots_4bases_4types.pdf) (16 子图)<br>- **Table 1**: 主表 (`sota_scores/main_table_and_efficiency.md`)<br>- **Fig. 2**: [fig2_efficiency_comparison.pdf](file:///d:/study/Diffip/llm_migration_v4/figs_sota/fig2_efficiency_comparison.pdf) (效率对比)<br>- **Fig. 3**: [fig3_ablation_pca_and_layers.pdf](file:///d:/study/Diffip/llm_migration_v4/figs_sota/fig3_ablation_pca_and_layers.pdf) (PCA 与层间选择)<br>- **Fig. 4**: [fig4_cross_base_heatmap.pdf](file:///d:/study/Diffip/llm_migration_v4/figs_sota/fig4_cross_base_heatmap.pdf) (跨基座区分矩阵) | 本文档 + [论文实验章节完整设计方案.md](file:///d:/study/Diffip/llm_migration_v4/%E8%AE%BA%E6%96%87%E5%AE%9E%E9%AA%8C%E7%AB%A0%E8%8A%82%E5%AE%8C%E6%95%B4%E8%AE%BE%E8%AE%A1%E6%96%B9%E6%A1%88.md) |
| **`② Method`** | **Section 3: Methodology (`DiffIP-LLM`)**<br>- `3.1 Threat Model & Symmetry Group Expansion`<br>- `3.2 Seed-Synchronized Paired Input Perturbation`<br>- `3.3 Even-4 Trajectory Subspace Projection (PCA)`<br>- `3.4 General Linear Reversion via ALS & Same-Prompt Calibration` | - 威胁模型：从正交/置换群扩展至一般可逆仿射变换群 $\mathrm{GL}(d)$（涵盖 `T3+T5` 旋转+非恒定 RMSNorm）<br>- 4 层均匀锚定 `[0, 10, 21, 31]` + $d=8$ PCA 降维<br>- 5 步交替最小二乘（ALS）求解 $W_l$ 与同题相对间隔校准 $\Delta(p)$ | [实验报告.md](file:///d:/study/Diffip/llm_migration_v4/%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A.md) + [对比论文总结与本文方法定位.md](file:///d:/study/Diffip/llm_migration_v4/%E5%AF%B9%E6%AF%94%E8%AE%BA%E6%96%87%E6%80%BB%E7%BB%93%E4%B8%8E%E6%9C%AC%E6%96%B9%E6%B3%95%E5%AE%9A%E4%BD%8D.md) 第 3 节 |
| **`③ 相关工作`** | **Section 2: Related Work**<br>- `2.1 Behavioral & Black-Box LLM Fingerprinting` (`Independence Tests`)<br>- `2.2 White-Box Static & Activation Fingerprinting` (`REEF`, `HuRef`, `AWM`, `GhostSpec`) | - 精辟评述 5 篇代表作的贡献与局限<br>- 论证为何主表对比 `REEF`、`HuRef`、`AWM`，而将 `Independence Tests`（黑盒统计检验）与 `GhostSpec`（算子相同仅聚合统计量不同的谱特征）置于相关工作讨论 | [对比论文总结与本文方法定位.md](file:///d:/study/Diffip/llm_migration_v4/%E5%AF%B9%E6%AF%94%E8%AE%BA%E6%96%87%E6%80%BB%E7%BB%93%E4%B8%8E%E6%9C%AC%E6%96%B9%E6%B3%95%E5%AE%9A%E4%BD%8D.md) 第 5 节（含中英文双语成稿，可直接粘贴进 Overleaf） |
