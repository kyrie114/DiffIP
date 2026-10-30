# FingerInv Table-2 攻击补测（LLM v3 面板）

日期：2026-10-30（Asia/Shanghai）。  
代码：`collect.py`、`score.py`；汇总：`results.json`；逐条打分：`scores/*.json`。  
特征 `feats/*.npz` **不提交**（由 `llm_migration_v3/.gitignore` 的 `feats/` 规则忽略，可本地重采）。

## 1. FingerInv Table 2 是什么

FingerInv（CVPR 2025）Table 2 按 **3 类攻击** 汇总鲁棒性验证，共 **22 个** model×attack 格子（Ours 成功率 22/22 = 100%；CycleDiff 18/22 ≈ 81.82%）。三类是：

| 类 | 论文设定（摘录） | 格子大致覆盖 |
|---|---|---|
| **Finetuning** | PS-DDPM 上 300/600/1000 iter；LDM 侧用 SD V1-5 / Delibrate / Realistic / Anything 等微调衍生 | 若干模型×微调 |
| **Pruning** | 幅值/结构剪枝强度 PN 2%–50%（PS-DDPM 常用 5%/10%；SD 到 30%/50%；Deci/Pixart 5%/10%/15%） | 若干模型×剪枝强度 |
| **Quantization** | float16 / bfloat16 | 若干模型×量化 dtype |

原文对象是 **扩散去噪网络**，不是 LLM。本目录把这三类迁移到 **v3 同一 LLM 面板**：victim = Qwen2.5-0.5B；suspect 默认叠在 **Instruct 微调权重**上再施加攻击（与 v3 `instruct_prune` 行同思路）。协议与 v3 扰动主表一致：32 prompts、SEED_SETS×3、N=128、SCALE=2e-3、PCA dims=8、全 4 均匀层采集；打分同时报 **all4** 与 **middle2=[1,2]**。

## 2. 我们测了什么（对应关系）

| FingerInv 类 | 本目录攻击名 | 实现 | 真假 |
|---|---|---|---|
| Finetuning | `fine_tune` | 直接用既有 v3 Instruct vs base（不重采） | 真 · 引用 |
| Pruning | `prune_dpoff` | v3 默认层丢弃（约 15%，DP 关） | 真 · 引用 |
| Pruning | `lprune_{5,10,15,30,50}` | 整层均匀丢弃，`n_drop=round(frac·24)`（最少 1） | 真 · 新测 |
| Pruning（额外） | `mprune_{5,10,15,30,50}` | 每层 Linear 全局 L1 幅值非结构化剪枝 | 真 · 新测（FingerInv 风格补强） |
| Quantization | `q_fp16` / `q_bf16` | 权重与前向均为该 dtype（CPU torch） | 真 · 新测 |
| Quantization（额外） | `q_int8dyn` | `torch.ao` 动态 int8（Linear） | 真 · 新测 |
| Quantization（额外） | `q_w8sim` / `q_w4sim` | **模拟**：按输出通道对称 RTN fake-quant，再反量化回 fp32 计算 | **模拟** · 新测 |

未测且不适用/本机做不到的：FingerInv 里扩散专用的「按 FID/SSIM 看生成质量」格子；真实 int4/GPTQ/AWQ 内核（无对应 CPU 部署栈，故用 `q_w4sim` 近似）。**置换 / 缩放** 不在 FingerInv Table 2 三类里，已在 v3 主表，本报告不重复。

## 3. 主结果（middle2，推荐读这一张）

设定：N=128，dims=8，seed set 0；`gap = sim − max_neg_mean`，负例 max mean = **0.144**；AUC 为攻击 32 prompts vs 全部负例 prompts 的合并比较；`perm_p` 为单侧置换检验（20000 次标签打乱）。

| attack | sim(seed0) | 3-set mean±std | gap | AUC | wins | perm p | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| fine_tune | 0.766 | 0.766±0.006 | **+0.622** | 0.989 | 32/32 | 5.0e-05 | 引用 |
| prune_dpoff | 0.749 | 0.749±0.006 | +0.605 | 0.988 | 32/32 | 5.0e-05 | 引用 · ≈lprune_15 |
| lprune_5 | 0.763 | 0.763±0.006 | +0.619 | 0.988 | 32/32 | 5.0e-05 | |
| lprune_10 | 0.761 | 0.760±0.006 | +0.617 | 0.988 | 32/32 | 5.0e-05 | |
| lprune_15 | 0.749 | 0.749±0.006 | +0.605 | 0.988 | 32/32 | 5.0e-05 | = prune_dpoff |
| lprune_30 | 0.740 | 0.740±0.005 | +0.595 | 0.988 | 32/32 | 5.0e-05 | |
| lprune_50 | 0.231 | 0.236±0.006 | **+0.087** | 0.678 | 32/32 | 3.4e-03 | **弱** |
| mprune_5 | 0.765 | 0.766±0.005 | +0.621 | 0.990 | 32/32 | 5.0e-05 | |
| mprune_10 | 0.750 | 0.751±0.005 | +0.606 | 0.989 | 32/32 | 5.0e-05 | |
| mprune_15 | 0.679 | 0.683±0.007 | +0.534 | 0.976 | 32/32 | 5.0e-05 | |
| mprune_30 | 0.366 | 0.363±0.004 | +0.222 | 0.838 | 32/32 | 5.0e-05 | 明显掉 |
| mprune_50 | 0.202 | 0.201±0.008 | **+0.058** | 0.640 | 30/32 | 3.2e-02 | **弱** |
| q_fp16 | 0.763 | 0.764±0.007 | **+0.619** | 0.988 | 32/32 | 5.0e-05 | ≈ fine_tune |
| q_bf16 | 0.580 | 0.592±0.021 | +0.436 | 0.920 | 32/32 | 5.0e-05 | 弱于 fp16 |
| q_int8dyn | 0.195 | 0.195±0.003 | **+0.051** | 0.631 | 32/32 | 4.7e-02 | **弱** · 真 int8 |
| q_w8sim | 0.767 | 0.768±0.006 | +0.623 | 0.988 | 32/32 | 5.0e-05 | **模拟** RTN-8 |
| q_w4sim | 0.309 | 0.306±0.004 | +0.165 | 0.785 | 32/32 | 5.0e-05 | **模拟** RTN-4 |

## 4. 对照表（all4）

负例 max mean = **0.235**。趋势与 middle2 一致，绝对 gap 更小。

| attack | sim(seed0) | gap | AUC | wins | perm p |
|---|---:|---:|---:|---:|---:|
| fine_tune | 0.758 | +0.522 | 0.962 | 32/32 | 5.0e-05 |
| prune_dpoff | 0.646 | +0.411 | 0.918 | 32/32 | 5.0e-05 |
| lprune_5 | 0.733 | +0.497 | 0.961 | 32/32 | 5.0e-05 |
| lprune_10 | 0.710 | +0.475 | 0.943 | 32/32 | 5.0e-05 |
| lprune_15 | 0.646 | +0.411 | 0.918 | 32/32 | 5.0e-05 |
| lprune_30 | 0.617 | +0.382 | 0.891 | 32/32 | 5.0e-05 |
| lprune_50 | 0.303 | +0.067 | 0.623 | 32/32 | 2.5e-02 |
| mprune_5 | 0.754 | +0.518 | 0.960 | 32/32 | 5.0e-05 |
| mprune_10 | 0.734 | +0.499 | 0.960 | 32/32 | 5.0e-05 |
| mprune_15 | 0.675 | +0.440 | 0.944 | 32/32 | 5.0e-05 |
| mprune_30 | 0.449 | +0.213 | 0.794 | 32/32 | 5.0e-05 |
| mprune_50 | 0.301 | +0.066 | 0.610 | 31/32 | 2.9e-02 |
| q_fp16 | 0.756 | +0.520 | 0.961 | 32/32 | 5.0e-05 |
| q_bf16 | 0.598 | +0.362 | 0.872 | 32/32 | 5.0e-05 |
| q_int8dyn | 0.289 | +0.053 | 0.594 | 32/32 | 5.9e-02 |
| q_w8sim | 0.757 | +0.521 | 0.961 | 32/32 | 5.0e-05 |
| q_w4sim | 0.401 | +0.166 | 0.739 | 32/32 | 5.0e-05 |

## 5. 诚实分析

1. **fp16 几乎和 fine_tune 一样强**（middle2 sim 0.763 vs 0.766，gap +0.619）。在本协议下，仅把 Instruct 权重/前向改成 fp16，不足以拆开指纹。
2. **bf16 明显弱于 fp16**（0.580 / gap +0.436），但仍远高于负例，AUC 0.920。CPU bfloat16 数值行为与 fp16 不同，不能把「半精度」当成一类。
3. **真动态 int8（`q_int8dyn`）是本批最弱的量化**：sim 0.195，gap 仅 +0.051，perm_p≈0.047（middle2）——与负例几乎擦边。这是 **真实** 量化，不是模拟。
4. **模拟 RTN-8（`q_w8sim`）很强、RTN-4（`q_w4sim`）中等偏弱**。模拟只改权重再 fp32 前向，**不能**外推成真实 int4/GPTQ 部署结论；只说明「权重量化噪声」量级对 DiffIP 的影响方向。
5. **层丢弃到 30% 仍稳**（lprune_30 sim 0.740）；**50% 崩**（0.231）。与 v3 默认 prune（≈15%）一致：中等结构剪枝可归因，极端剪枝不行。
6. **幅值剪枝更敏感**：mprune_15 已掉到 0.679，mprune_30→0.366，mprune_50→0.202。非结构化剪枝比同比例层丢弃更伤表示。
7. **middle2 相对 all4**：强攻击（微调/轻剪/fp16）gap 更大；弱攻击（int8dyn / 50% 剪）两边都弱，middle2 没有「救」回来。
8. **与 FingerInv 22 格的关系**：三类都覆盖到了；格子数上我们用 **强度梯度 + 真/模拟量化变体** 代替扩散模型清单，不是逐格复现 Table 2。置换/缩放/蒸馏另见 v3 主表与 `improve/蒸馏鲁棒性.md`。

## 6. 复现

```bash
cd llm_migration_v3/fingerinv_attacks
# 需已有 ../feats/pert_victim.npz 与 ../scores/
python collect.py q_fp16 q_bf16 q_int8dyn q_w8sim q_w4sim \
  lprune_5 lprune_10 lprune_15 lprune_30 lprune_50 \
  mprune_5 mprune_10 mprune_15 mprune_30 mprune_50
python score.py   # 或显式传入未打分的名字；会刷新 results.json
```

`feats/` 体积大（每个约 168MB），**不入库**；提交的是代码、`scores/`、`results.json` 与本报告。
