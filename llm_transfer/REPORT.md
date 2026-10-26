# DiffIP → LLM 迁移实验报告（收尾文档）

日期：2026-10。目标：评估 DiffIP（扩散模型表示指纹）方法迁移到大语言模型
谱系判定的可行性。所有实验使用 conda env `diffip-llm`
（Python 3.12，torch CPU，transformers），模型经
`HF_ENDPOINT=https://hf-mirror.com` 下载。

## 协议

- 表示采集：`diffip.llm.LLMRepresentationCollector`，hook 抓取 decoder 层
  hidden states，逐层 attention-mask 加权 mean pooling → `[L, N, D]`
- 采样：perturbation 模式（同 prompt 输入 embedding 加高斯噪声），
  **配对种子**（victim 与 suspect 必须用同一 seed，对应扩散方法的
  "seed order must remain consistent"）
- 工作点：`perturbation_scale=2e-3`，`seed=123`，L=4 层，N=8
- 判定：`DiffIPMetric.compare()`（独立 PCA-20 → 逐步 Procrustes 回退 → DP 时序对齐）

## 主要结果

### 1. 尺度扫描（scripts/sweep_perturbation.py，配对种子修复后 scripts/sweep_modes.py）

- 未配对种子时距离被采样噪声主导，响应曲线呈 0/1 悬崖（错误协议的假象）
- 配对后 perturbation 模式曲线单调：≤3e-3 权重噪声 similarity≈1.0，
  1e-1（无关代理）≈0.50 → 工作点 scale=2e-3

### 2. 真实模型对验证（scripts/validate_real_pair.py）

| 配对 | similarity |
|---|---|
| SmolLM2-135M 自比 | 1.000 |
| SmolLM2-135M ↔ Instruct（真实 SFT，正例） | 0.505 |
| SmolLM2-135M ↔ Qwen2.5-0.5B（跨家族，负例） | 0.323 |

排序正确；跨维度（D=576 vs 896）与跨层数 DP 对齐均可用。

### 3. 层数/样本数消融（scripts/ablate_layers_samples.py）

- L：4→16 裕度 0.13→0.18（不单调，计算量 ×16）
- N：4 或 16 均恶化（N=16 时正例塌到 0.04）；N=8 即上限
- 结论：配置族天花板 ~0.18 similarity（距离空间 ~25-35% 分离）

### 4. 混合采样（scripts/ablate_hybrid.py）——否定结果

prompt 池拼入同一样本轴后正负例全部归零：异质 prompt 云违反配对
Procrustes 的"两云只差一个线性映射"假设。划定了方法适用边界。

### 5. 分 prompt 评分聚合（scripts/per_prompt_eval.py）——否定结果

逐 prompt 独立评分后聚合：距离空间正例均值 2.21 vs 负例 2.87，
逐 prompt 分布重叠严重，只降方差、不提可分性。

### 6. ROC 标定（scripts/roc_calibration.py）

4 模型（SmolLM2/Qwen2.5-0.5B 各 base+Instruct）、2 正例对 + 4 负例对、
48 个逐 prompt 距离样本：

- **AUC = 0.678**；最佳阈值 distance ≤ 0.737（TPR=0.50, FPR=0.125）
- 距离绝对尺度家族依赖：Qwen 对 0.1-0.7，SmolLM2 对 1-4；
  SmolLM2 的 SFT 位移与跨家族差异同量级 → **全局阈值跨家族不可迁移**，
  实用需家族内相对标定

## 总结论

1. DiffIP 结构上可迁移至 LLM：排序自比 > 真实 SFT > 跨家族始终成立，
   跨维度/跨层数由独立 PCA + DP 对齐天然支持。
2. 裕度天花板 ~0.18 由方法核心决定（异模型层表示云不满足
   "只差一个线性映射"的回退假设），采样策略无法突破。
3. 单一全局阈值不可用；家族内相对标定（参照模型归一化）是最低改造成本，
   更根本的改进需偏离原方法（非线性/局部对齐），应另起课题。

## 脚本索引

| 脚本 | 用途 |
|---|---|
| `scripts/sweep_perturbation.py` | perturbation_scale 扫描 |
| `scripts/sweep_modes.py` | 扰动 vs 多 prompt 采样对比（响应曲线） |
| `scripts/validate_real_pair.py` | 真实微调对验证 |
| `scripts/ablate_layers_samples.py` | 层数/样本数消融 |
| `scripts/ablate_hybrid.py` | 混合采样消融（否定结果） |
| `scripts/per_prompt_eval.py` | 分 prompt 评分聚合 |
| `scripts/roc_calibration.py` | ROC 标定 |
