# llm_migration_v4 执行计划（待审批）

> 状态：**等待批准**。批准前不会启动任何长时间任务。
> 背景见 `PROGRESS.md`（已发生的过程与决策记录）。

## 一、当前状态

- v4 脚本初版已写完（`common.py / collect.py / score.py / gen_teacher.py / train_student.py`）。
- 首次试跑的三个任务**全部启动即失败**，现已确认无残留进程：

| # | 故障 | 根因 | 修复 |
|---|---|---|---|
| 1 | `collect.py main` 立即崩溃 | `common.py: build_main` 中 `from transforms import ...` 在 `sys.path.insert` **之前**执行 | 把 path 插入 + 导入提到模块顶部（一行位置调整） |
| 2 | `gen_teacher.py` 立即崩溃 | 本地 Llama-3.2-1B/3B 的 `tokenizer_config.json` **没有 chat_template**（已核实 1B/3B/Llama-2-7b 三个模型的本地拷贝均缺失） | `common.py` 内置 Llama-3.2 官方 chat 模板字符串，加载后 `tok.chat_template = ...` 显式设置（gen_teacher / train_student 共用） |
| 3 | `train_student.py control` 立即崩溃 | 同 #2 | 同 #2 |

注意：**主线采集不受 chat_template 影响**（只用原始 prompt 的 token id），故障 #1 修好即可跑；
#2/#3 只阻塞蒸馏线。

## 二、实验设计（与 PROGRESS.md 一致，未变）

- **主线**：victim = Llama-2-7b-hf；正例 = Llama-2-7b-chat-hf（官方 SFT）+ Chinese-llama-2-7b（独立衍生）；
  伪装 = weight_perm / rep_perm / scale / prune_dpoff；难负例 = 同底座 3 个剪枝变体；易负例 = Qwen2.5-7B / Llama-3-8B / internlm2-7b。
- **蒸馏线**：victim = Llama-3.2-1B；学生 = 1B 基座分别训 3 epoch（distill：Llama-3.2-3B 教师回答；control：Alpaca 原始回答，同预算）；
  负例 = Llama-3.2-3B / Sheared-LLaMA-1.3B / Qwen2.5-7B / Llama-2-7b-hf（跨维度）。
- **协议**：与 v3 完全一致（SCALE 2e-3、配对种子 ×3、32 prompts、N≤128、4 层、PCA d∈{8,20} + 逐层 ALS、DP 关）。
- **资源**：全 CPU（48 核三分：采集 16 / 生成 16 / 训练 12 线程）；GPU 不碰（qvh 任务）；强制离线；磁盘 ~10GB（feats）；内存峰值 <25GB。

## 三、执行阶段

### P0 修复 + 冒烟（~15 分钟，无长任务）
1. 修 `common.py`（#1 导入顺序；#2 内置 Llama-3.2 chat 模板 + 助手函数），改 `gen_teacher.py`/`train_student.py` 调用。
2. 新增 `smoke_test.py`，三项检查，全部通过才允许进入 P1：
   - **S1 采集**：Llama-3.2-1B 上 1 prompt × 1 seed set × 128 样本 × 4 层前向 → 形状 [4,128,2048]，无 NaN，与 v3 协议逐样本一致性断言；
   - **S2 教师生成**：Llama-3.2-3B 用内置模板生成 1 条回答（≤48 token）→ 非空；
   - **S3 学生训练**：Llama-3.2-1B 用内置模板跑 2 步训练（1 条数据）→ loss 为有限值且下降不报错。

### P1 三条并行任务（批准后启动）

| 任务 | 命令 | 线程 | 预计 | 日志 | 完成标志 |
|---|---|---|---|---|---|
| A 主线采集（11 个模型逐一跑） | `python collect.py main` | 16 | ~6.5h | `collect_main.log` | `feats/main_*.npz` ×10 |
| B 教师生成（1720 条） | `python gen_teacher.py` | 16 | ~4h | `gen_teacher.log` | `data/teacher_data.json` 完整 |
| C control 学生训练 | `python train_student.py control 3` | 12 | ~1.5h | `train_control.log` | `ckpt/student_control/` |

### P2 链式任务（前置完成后启动，各需一次确认触发）
- B 完成 → `python train_student.py distill 3`（~1.5h）
- A 完成 → `python score.py main`（~30min，多进程 ALS）
- C+distill 完成 → `python collect.py distill victim student_distill student_control neg_llama3_2_3b neg_sheared_1_3b neg_qwen2_5_7b neg_llama2_7b`（~1.7h）→ `python score.py distill …`

### P3 收尾
- 写 `实验报告.md`（主线表 + 蒸馏线结论 + 与 v3 0.5B 结果对照 + 诚实的局限）
- git 提交（脚本 + PROGRESS.md + PLAN.md + 报告 + 汇总表）并推送 fork

## 四、时间线（批准后 T0 起）

| 时刻 | 事件 |
|---|---|
| T0+0.25h | P0 冒烟通过 |
| T0+1.5h | C 完成（control 学生） |
| T0+4h | B 完成（教师数据）→ distill 学生开训 |
| T0+5.5h | distill 学生完成 → 蒸馏线采集开跑 |
| T0+6.5h | A 完成（主线采集）→ 主线评分开跑 |
| T0+7h | 主线结果表 |
| T0+8.2h | 蒸馏线结果表 |
| T0+9h | 报告 + 提交推送，全部结束 |

## 五、风险与中止判据（出现即停下报告，不硬闯）

| 风险 | 判据 | 动作 |
|---|---|---|
| 单模型采集超慢 | 任一 7B 模型单 prompt 超过 3 分钟 | 查 CPU 竞争，必要时降到 8 线程重排 |
| 教师生成进度异常 | T0+2h 时 train 段 <40% | 降 max_new_tokens 或 BS 后重跑（数据文件断点续写） |
| 训练发散 | loss >10 或 NaN | 停训，检查数据/模板拼接，不强行降 lr 掩盖 |
| 采集数值异常 | S1 之外出现 NaN/全零层 | 该模型隔离，其余继续 |
| 磁盘 | feats 超过 15GB | 不应发生（预估 8GB），若发生先查异常文件 |

## 六、验收标准

1. `summary_main.json`：fine_tune / derivative_cn / weight_perm / rep_perm / scale / prune_dpoff
   在 N=128,d=8 下显著高于全部跨家族负例；同底座剪枝负例的位置符合“真实同源”预期；
2. `summary_distill.json`：student_distill 相对 victim 基座与 student_control 的 margin 给出明确符号
   （认得出 or 认不出，二者都是有效结论）；
3. 报告 + 代码全部入库并推送；
4. 全程未使用 GPU、未从 HF 下载任何权重。

## 七、请审批的决策点

1. **按本计划执行**（P0→P1→P2→P3，通宵跑，明早 ~06:00 全部出结果）；
2. 或**缩小主线负例面板**（去掉 neg_llama3_8b 与 neg_internlm2_7b，省 ~1.3h，主线 ~5h 完成）——牺牲跨家族覆盖面；
3. 其他调整（改模型 / 改预算 / 改时间窗）。
