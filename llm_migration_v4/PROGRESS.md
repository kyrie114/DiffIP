# llm_migration_v4 — 任务过程记录

日期：2026-10-30（周五）晚启动，预计 10-31 早上出全部结果。
目标：把 v3 协议（配对种子扰动采样 + 逐 prompt 指纹 + PCA/ALS，无 DP）**放大到 7B 规模**，
用本地模型库 `/data/luo_lixuan/llms` 重跑，回答两个问题：
1. 主协议在 7B 上对**真实谱系**（官方 SFT、独立衍生、剪枝变体、跨家族）的判别力如何？
2. **蒸馏**（序列级 KD，异词表/异规模）留下的 hidden 几何指纹能否被认出？（v1 判负，v3 协议下未重验）

## 为什么换模型、为什么这样配

- v3 在 0.5B 上做出 0.52 margin / 32/32 全可分，但 victim 只有 0.5B，负例面板弱
  （SmolLM2-135M 等小模型）；ROC 标定又发现**距离尺度家族依赖**——需要 7B 级、
  同底座真实衍生 + 跨家族 7B 负例来检验协议是否真正可迁移。
- 本机 4×RTX 5090 被 `qvh_b96_llr` 任务占满（每卡仅剩 ~5GB，3B 都放不下），
  **全部走 CPU**（48 核，torch 16/16/12 线程三分）。基准实测：
  Llama-3.2-1B 128 样本 ≈ 5.3s/prompt，Qwen2.5-7B ≈ 23.5s/prompt → 7B 模型采集 ~38min/个。
- 模型一律取自本地库，**不做任何下载**（HF_HUB_OFFLINE=1 强制离线）。
- 环境：`diffip-llm`（torch 2.14.1+cpu，transformers 5.20.2）——冒烟验证通过。
- 蒸馏线注意：本地 Llama-3.2-1B/3B 是 **base 模型**（无指令微调，chat 模板对其无效，
  会复述+乱码），教师生成与双学生训练一律用 **v1 的纯 Alpaca 完成格式**
  `### Instruction:\n{ins}\n\n### Response:\n`，不用 chat 模板。
- internlm2-7b 含自定义代码，`load()` 统一传 `trust_remote_code=True`（离线加载本地
  modeling 文件，无网络依赖）。

## 两条实验线

### 主线（Llama-2-7b 家族，victim = Llama-2-7b-hf）

| 角色 | 模型 | 说明 |
|---|---|---|
| victim | Llama-2-7b-hf | 基座 |
| fine_tune | Llama-2-7b-chat-hf | 官方 chat SFT（对应 v3 的 Instruct 正例） |
| derivative_cn | Chinese-llama-2-7b | 独立真实衍生（第二个正例） |
| weight_perm | chat + 维度置换（fold=False，seed=0） | 白盒伪装，噪声先加后映射 x@P |
| rep_perm | fine_tune 特征后置换（seed=1） | 特征空间伪装（免采集，score 时派生） |
| scale | fine_tune 特征各向异性缩放（seed=0） | 特征空间伪装（免采集） |
| prune_dpoff | chat 剪 4 层（prune_spec） | 结构变换（DP 关闭） |
| neg_pruned_sparsegpt / wanda / gblm | 同底座剪枝变体 | **难负例**：真实同源（剪枝），应得高分 |
| neg_qwen2_5_7b / llama3_8b / internlm2_7b | 跨家族 7B/8B | 易负例，应得低分 |

### 蒸馏线（Llama-3.2，victim = Llama-3.2-1B）

| 角色 | 模型 | 说明 |
|---|---|---|
| victim | Llama-3.2-1B | 基座 |
| student_distill | 1B 学生 | 在 **Llama-3.2-3B 教师**回答上序列级 CE 蒸馏（1600 指令 ×3 epoch，chat 模板，答案段计损） |
| student_control | 1B 学生 | 同指令/同预算，训练文本换成 **Alpaca 原始回答**（关键对照：区分"学过教师" vs "只是被训练过"） |
| neg_llama3_2_3b | Llama-3.2-3B | 同家族更大（教师本身） |
| neg_sheared_1_3b | Sheared-LLaMA-1.3B | 不同血统、相近规模 |
| neg_qwen2_5_7b | Qwen2.5-7B | 跨家族跨规模 |
| neg_llama2_7b | Llama-2-7b-hf | 跨家族跨维度（4096 vs 2048，检验独立 PCA 路径） |

教师数据：沿用 v1 的 Alpaca 切分（1600 train / 120 heldout，`llm_migration/exp_distill_v2/data/`），
3B 教师重新生成回答（**纯 Alpaca 完成格式**，因本地 Llama-3.2-3B 是 base 模型；贪心，
max_new_tokens=64，BS=32）。

## 统一协议（与 v3 完全一致）

- SCALE=2e-3，配对种子 SEED_SETS=[123,1123,2123]，32 prompts（同 v3 探针集），
  N_MAX=128（评分取前 N ∈ {8,16,20,32,64,128}），4 个均匀层，attention-mask 均值池化
- 评分：逐 (prompt, seed set) 独立指纹 → PCA 降 d ∈ {8,20} → 逐层 ALS 回退
  （`diffip.preprocessing.project_sequences` + `diffip.reversion.fit_step_alignment`）
  → distance_mean / similarity = exp(-Σd/L)，DP 关闭
- 采集实现：`output_hidden_states=True` 批量取隐层（无 hook），失败自动降级 batch=1
  （`perturb_batch` 要求 `[1,T,D]` 输入，见下方事故记录 #1）

## 运行状态（截至 2026-10-30 ~19:40，用户要求暂停）

三条任务 19:18 启动，19:40 应用户要求全部停止。停止时磁盘零产物（无 npz / 无 checkpoint /
teacher 文件未重写），重启即从头跑，损失仅 ~15 分钟算力。

| 任务 | 命令 | 线程 | 实测速率（qvh 占 ~18 核下） | 预计总耗时 | 停时进度 |
|---|---|---|---|---|---|
| A 主线采集（11 个模型） | `python collect.py main` | 16 | ~115s/prompt（7B） | ~12h | victim 7/32 prompts |
| B 教师生成（1720 条） | `python gen_teacher.py` | 16 | ~13min/batch（3B, BS=32） | ~11.7h | 32/1600 条 |
| C control 学生训练 | `python train_student.py control 3` | 12 | ~35s/step（1B, BS=8） | ~5.8h | step 25/600 |

恢复：重跑上表三条命令即可（A 对已有 npz 自动 skip）。

后续链（衔接需人工触发一次）：
- B 完成后 → `python train_student.py distill 3`（~5.8h，用 B 产出的新 teacher 数据）
- A 完成后 → `python score.py main`（多进程 ALS，~30min）
- C/distill 训练完成后 → `python collect.py distill victim student_distill student_control neg_llama3_2_3b neg_sheared_1_3b neg_qwen2_5_7b neg_llama2_7b`（~3h）→ `python score.py distill ...`

## 检查进度的方法

```bash
cd llm_migration_v4
tail -2 collect_main.log gen_teacher.log train_control.log
ls feats/        # 每完成一个模型多一个 {line}_{name}.npz（断点续采：已存在自动 skip）
ls ckpt/         # student_distill / student_control 训完才出现
ls data/         # teacher_data.json 的 train 条数随 B 增长（每 256 条落盘一次）
```

## 关键决策记录

1. **CPU 而非 GPU**：GPU 被 qvh 任务占用且不可动；7B CPU 采集 ~38min/个，通宵可完成。
2. **主线选 Llama-2-7b 家族**：本地库中唯一拥有"官方 SFT + 独立衍生 + 3 个剪枝变体"
   完整谱系面板的家族；难负例（同底座剪枝）是 v3 没有的新维度。
3. **蒸馏学生 1B**：7B 学生 CPU 训练要 1-2 天；1B 与 3B 教师同家族（Llama-3.2），
   词表相同，蒸馏信号纯净，且当天可出结果。
4. **weight_perm 语义与 v3 逐位一致**：P=permutation(seed=0)，fold=False，
   输入噪声先加后 x@P（等价于 x[..., P.argmax(0)]）；RMSNorm γ 仅重排。
5. **rep_perm/scale 免采集**：特征空间派生（perm seed=1 / scale seed=0），
   与 v3 `score.py` 的 SUS 构造一致，少跑两次 7B 采集。
6. **不碰 GPU 任务、不装包**：全程 diffip-llm 环境（torch 2.14.1+cpu，transformers 5.20.2），
   强制离线加载本地模型。

## 事故与修复记录（2026-10-30）

1. **`perturb_batch` 形状 bug（我的，已修）**：v4 `collect.py` 里 `base = emb(ids)[0]`
   把 batch 维砍掉，`perturb_batch` 输出变成 2D `[n*T, D]`，模型把它当成
   "n*T 条长度为 D 的序列"前向，在 transformers 5.20 上表现为 RoPE 广播报错
   （32 heads vs 64 head_dim），一度被**误诊为"transformers 5.20 对 Llama 架构
   inputs_embeds 前向损坏"**。对照 v3 源码后确认 v3 的 `emb(ids)` 返回 `[1,T,D]`
   （带 batch 维）没有此问题，**v3 的 0.52 结果不受影响**；5.20 环境经 3D 输入
   复现验证完全正常。修复：所有调用点保留 `[1,T,D]`（去掉 `[0]`）。
2. **Llama-3.2 base 模型 + chat 模板（设计修正）**：本地 Llama-3.2-1B/3B 为 base
   权重，喂 chat 模板会复述问题+输出乱码（纯文本续写完全正常，已对照验证）。
   蒸馏线全部改用 v1 纯 Alpaca 完成格式，与 v1 协议对齐。
3. 冒烟（smoke_test.py，diffip-llm 环境，27s）：S1 采集 [4,128,2048] 无 NaN 且确定性
   重跑逐位一致；S2 教师回答 `Paris\n\n### Explanation:...`；S3 学生 2 步 loss 有限。

## 文件清单

```
llm_migration_v4/
├── common.py           # 协议常量 + 模型注册表 + 采集原语（perturb_batch / pooled_layers）
├── collect.py          # 特征采集（main / distill 两线，断点续采）
├── score.py            # ALS 评分 + 汇总（含同底座/跨家族分组统计、逐 prompt 胜负）
├── gen_teacher.py      # 3B 教师回答生成（输出 data/teacher_data.json）
├── train_student.py    # 1B 学生训练（distill / control）
├── PROGRESS.md         # 本文件
├── feats/  ckpt/  data/  scores/   # 本地产物（.gitignore，不入库）
└── *.log                            # 任务日志（不入库）
```
