# llm_migration_v4 交接说明

> 更新日期：2026-11-01。本实验**已完成并提交**（HEAD `09cc822`，工作区干净）。
> 一句话状态：v3 协议（配对种子扰动采样 + 逐 prompt 指纹 + PCA/逐层 ALS，无 DP）放大到 7B 规模重跑完毕，
> 主线 12 模型 + 蒸馏线 7 模型评分全部落盘，最终报告在 `实验报告.md`。

## 1. 产物与位置

| 产物 | 位置 | 说明 |
|---|---|---|
| **最终报告** | `实验报告.md` | 设计 / 执行记录（含全部事故）/ 两线结果表 / v3 对照 / 小结 / 复现命令 |
| 过程记录 | `PROGRESS.md` | 时间线、事故与修复、关键决策 |
| 执行计划 | `PLAN.md` | 审批决策 + 风险表（internlm2 隔离即依据其中风险行） |
| 管外视角分析 | `analysis.py` | 结构变体、蒸馏对照等补充视角（读 `scores/*.json`；`V4_LAYER_SUBSET=mid2` 读 `scores_mid2/`） |
| 评分明细 | `scores/{main,distill}_<pair>.json` | 每模型 1152 行（32 prompt × 3 seed set × 6 N × 2 dims），含 `similarity` / `distance_mean` |
| 汇总表 | `scores/summary_{main,distill}.json` | `by_nd: dims→N→pair→[3 个 seed set 的均值]` |
| 特征 | `feats/*.npz` | 17 个（主线 10 + 蒸馏线 7），~11GB，`.gitignore` 不入库；每个含 `pert [32,3,4,128,D]` + `layers4` |
| 学生模型 | `ckpt/student_{distill,control}/` | Llama-3.2-1B 微调产物 + `train_{mode}.json` 训练日志 |
| 教师数据 | `data/teacher_data.json` | 1720 条（1600 train + 120 heldout），Llama-3.2-3B base 贪心生成，纯 Alpaca 格式 |
| 日志 | `*.log` / `*.log.1030` / `*.log.1031` | 任务与诊断日志，不入库；internlm2 诊断证据在 `diag_internlm2_nan.log` / `diag_internlm2_attn.log` |

注意：`rep_perm` / `scale` 两个伪装**没有独立 npz**——它们在评分时由 `fine_tune` 特征在内存中派生（`score.py` 的 `_DERIVED` 缓存），所以 `feats/` 里看不到它们。

## 2. 复现

```bash
conda activate diffip-llm    # torch 2.14.1+cpu, transformers 5.20.2, protobuf 6.33.6
export HF_HUB_OFFLINE=1       # 强制离线；模型一律取本地库 /data/luo_lixuan/llms
cd llm_migration_v4
python smoke_test.py          # ~30s，S1 采集确定性 / S2 教师 / S3 学生训练
python collect.py main        # 11 个模型 ~10h（8 线程，断点续采：已有 npz 自动跳过）
python score.py main          # ~15min（V4_SCORE_WORKERS 可调，默认 8）
python gen_teacher.py         # 3B 教师 1720 条 ~33min
python train_student.py distill 3    # ~4.7h
python train_student.py control 3    # ~5.8h
python collect.py distill victim student_distill student_control \
    neg_llama3_2_3b neg_sheared_1_3b neg_qwen2_5_7b neg_llama2_7b
python score.py distill victim student_distill student_control \
    neg_llama3_2_3b neg_sheared_1_3b neg_qwen2_5_7b neg_llama2_7b
```

确定性已验证：采集与评分重跑均**逐位一致**（11-01 全天两条线各重跑一次核对过）。
完整命令与耗时见 `实验报告.md` "复现" 节。

## 3. 接手必须知道的坑（非显然信息）

1. **internlm2-7b 已隔离，别再直接采**。它在 transformers 5.20 CPU + trust_remote_code 下深层激活发散为 NaN
   （权重全部有限；NaN 起始层随 run 漂移 L2/L4/L12/L21；第 1 层激活幅值 ~4800；SDPA 重采在 prompt 2 仍失败）。
   其 npz 已删除，跨家族面板 = Qwen2.5-7B + Llama-3-8B。如需该模型：换 GPU 或 transformers 4.x 环境重试，
   重采后跑 `python score.py main neg_internlm2_7b` 单独补分即可（summary 需全量重跑合并）。证据链见报告 4.3。
2. **`score.py` 的 THP 修复不能删**（`mallopt` 阈值 + `prctl(PR_SET_THP_DISABLE)`）：没有它评分会让全机
   compaction 风暴卡死（`kcompactd` 100%、worker 100% 零产出，11-01 19:47 事故）。并发用 `V4_SCORE_WORKERS` 调。
3. **`common.py load()` 的 rope 补丁只对 trust_remote_code 自定义 config 生效**。在 5.20 上对标准 Llama
   （Sheared-LLaMA / Llama-2 系）回写 `rope_scaling=None` 会抹掉 `rope_parameters` 导致加载崩溃——已踩过。
   同文件的 `ATTN_IMPL` 是按模型目录名做的注意力实现覆盖表（internlm2→sdpa），新增怪模型从这里加。
4. **蒸馏线两个负例是主线 npz 的复制**（`distill_neg_llama2_7b.npz` = `main_victim.npz`，
   `distill_neg_qwen2_5_7b.npz` = `main_neg_qwen2_5_7b.npz`，协议一致、确定性采集）。动主线这两个 npz 会连带影响蒸馏线结果。
5. **`collect.py` 有逐 prompt 有限性校验**：它中止（"non-finite features — aborting"）说明**模型/环境数值问题**，
   不是采集代码 bug——按 PLAN 风险表先诊断模型（权重有限性 → 干净前向逐层有限性 → 换精度/注意力实现），再决定隔离。
6. **指标口径**：主指标是 `distance_mean`（逐层 ALS 平均 L2），`similarity = exp(−Σd/L)`；表里 "sim±std / dist"
   的 ± 是 3 个 seed set 的 std，两者都是跨 32 prompt 的均值。跨模型比较看排序，不看绝对值（阈值家族/规模依赖，v2 已证）。
7. **GPU 被 `qvh` 任务占着，别动**；本实验全程 CPU（48 核）。机器共享，长任务用 8 线程档（`V4_THREADS` / `V4_SCORE_WORKERS`）。
8. 分词器坑：transformers 5.20 读 sentencepiece `.model` 需要 **protobuf**（已装 6.33.6）；缺了会静默退回 tiktoken 然后崩溃。
   `load_tok` / `load` 一律带 `trust_remote_code=True`。

## 4. 结果速览（详情与解读见 `实验报告.md` 第 4/5/8 节）

**主线（victim = Llama-2-7b-hf；N=128, d=8，跨 prompt 均值）**

| 谱系位置 | similarity | 结论 |
|---|---|---|
| 同底座结构变体（3 个，正例；稀疏权重剪枝） | 0.53–0.56 | 稀疏剪枝比 SFT 更像基座（与 v3 层剪枝排序反转，但剪法/比例/规模因素不可分离） |
| SFT + weight/rep/scale 伪装 + chat剪枝 | 0.357–0.365 | 三种线性伪装全部隐形（互相噪声级不可分） |
| 独立衍生 Chinese-llama | 0.311 | 小 N 下与负例地板重叠更大（N=20 仅 20/32 胜跨家族） |
| 跨家族（Qwen2.5-7B / Llama-3-8B） | 0.254 | 地板稳定（v3 的 0.235–0.24 几乎不变） |

**蒸馏线（victim = Llama-3.2-1B）**：负结果。student_distill vs student_control 差异 ≤0.003（噪声内、方向不稳定），
教师本身 0.591 与无关负例无差别——**"被蒸馏过"在 N≤128 预算下不可检测**。

**核心结论**：协议可迁移到 7B、相对谱系排序可靠；绝对 similarity 随规模压缩（fine_tune 0.76→0.36，margin 0.52→0.11），
**跨规模可迁移排序、不可迁移阈值**，实用判定必须家族内 + 规模内标定。

**口径更新**：sparsegpt/wanda/gblm 改归正例"同底座结构变体"（代码键名仍是 `neg_pruned_*`，输出显示名为"结构变体_*"）；胜负只对跨家族负例计。prune_dpoff 是整层删除，三个结构变体是非结构化稀疏剪枝，两者不同。

**中间 2 层重评分（待跑）**：`V4_LAYER_SUBSET=mid2 python score.py main`（及 distill），输出 `scores_mid2/`。mid2 = 4 个存储层中的第 1、2 个，不等于 v3 的全层中间 2 层。结果尚未运行。

## 5. 后续可做（按性价比排序）

1. **补第三个跨家族易负例**（internlm2-7b）：需 GPU 或 transformers 4.x 环境，见坑 #1。
2. **蒸馏线大 N 复核**：distill−control 差 ~0.003 ≈ 1σ，N>128 或第 4 个 seed set 可验证是否出现微弱可分性（预算：N=256 约 2× 采集 + 评分时间）。
3. **7B 蒸馏学生**：1B 学生结论外推到 7B 需验证；CPU 1–2 天 / GPU 约 1h。
4. **跨 run / 跨机方差**：目前每模型单 run（确定性只验证了同机重跑）；论文级声明需 ≥1 次异构环境复跑。
5. **7B 面板 ROC 标定**：v2 的 ROC 方法（`llm_transfer/roc_calibration.py`）在 v4 面板上重做，给"家族内相对阈值"一个定量版。
6. **DP-on 对比**：需要全层数据（4 层→32 层），采集成本 ×8，优先级低。

## 6. 仓库地图（本实验在整个 DiffIP 项目中的位置）

```
DiffIP/
├── diffip/                # 核心库：preprocessing(PCA/投影) reversion(ALS) metric dynamic _validation
├── docs/LLM迁移实验导读.md  # 全项目导读（接手先读这个）
├── llm_migration/         # v1：方法探索（exp_distill_v2 蒸馏数据/协议源头、白盒对比等）
├── llm_transfer/          # 早期原型：ROC 标定、采样策略扫描
├── llm_migration_v2/      # v2：ROC 标定 + 负例面板（"距离尺度家族依赖"结论出处）
├── llm_migration_v3/      # v3：0.5B 完整协议（0.52 margin / 32/32，本 v4 的协议模板）
└── llm_migration_v4/      # 本目录：7B 规模重跑（已完成）
```

相关历史提交：`09cc822`（v4 收尾）← `100cd8b`（v4 首提交）← `88868be`（v3 failure_diag）← …

## 7. 当前状态核对清单

- [x] 主线 10 + 蒸馏线 7 npz 全有限（逐文件 `np.isfinite` 校验过）
- [x] 评分两线各重跑一次，数值逐位一致
- [x] `scores/` 21 个 json（19 明细 + 2 汇总）完整
- [x] 报告 / PROGRESS / PLAN / 本文件入库；大产物（feats/ckpt/data/scores、日志）按 .gitignore 不入库
- [x] internlm2-7b 隔离已按 PLAN 风险表执行并记录
