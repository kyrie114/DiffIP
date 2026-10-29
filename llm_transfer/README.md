# llm_transfer — 另一台机器上的 DiffIP→LLM 探索线

用官方 `LLMRepresentationCollector` + `DiffIPMetric.compare()`（PCA-20 → Procrustes → 可选 DP）做扰动尺度扫描、真实 SFT 对验证、层/样本消融、hybrid 否定与 ROC（AUC 0.678）。工作点：`perturbation_scale=2e-3`，N=8，L=4。

完整数字见 [`REPORT.md`](REPORT.md)。与本 bot「ALS + 平均 L2」严格重跑的**区别与联系**见 [`docs/LLM迁移实验导读.md`](../docs/LLM迁移实验导读.md)（两条线共享配对扰动思想，判定公式与样本量不同，数字不可直接横比）。
