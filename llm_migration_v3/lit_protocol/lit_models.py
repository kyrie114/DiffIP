"""Model panel for the literature-protocol batch. label: pos = derived from Qwen2.5-0.5B
(victim), neg = not derived (hard negative)."""
MODELS = {
 # positives: real public derivatives (REEF ICLR'25 Table 1 / HuRef NeurIPS'24 Table 2-3 style)
 "coder_base":   ("Qwen/Qwen2.5-Coder-0.5B", "pos", "continued pretraining on code (Qwen2.5-Coder report: ~5.5T tokens)"),
 "coder_inst":   ("Qwen/Qwen2.5-Coder-0.5B-Instruct", "pos", "continued PT + post-training"),
 "ita_sft":      ("ReDiX/Qwen2.5-0.5B-Instruct-ITA", "pos", "community SFT (Italian) on Instruct"),
 "vikhr_sft":    ("Vikhrmodels/Vikhr-Qwen-2.5-0.5b-Instruct", "pos", "community SFT (Russian) on Instruct"),
 "hammer_fc":    ("MadeAgents/Hammer2.0-0.5b", "pos", "function-calling FT on Instruct"),
 "abliterated":  ("huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3", "pos", "refusal-direction weight editing (abliteration)"),
 "nuextract":    ("numind/NuExtract-1.5-tiny", "pos", "extraction FT on base"),
 "r1_draft":     ("alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B", "pos", "SFT on DeepSeek-R1 outputs"),
 "slerp_merge":  ("Youlln/ECE.EIFFEIL.ia-0.5B-SLERP", "pos", "SLERP weight merge (mergekit)"),
 # hard negatives: same vendor / near-identical architecture, independent pretraining
 "qwen2_inst":   ("Qwen/Qwen2-0.5B-Instruct", "neg", "Qwen2 generation post-trained; different pretrain from victim"),
 "qwen3_base":   ("Qwen/Qwen3-0.6B-Base", "neg", "same vendor, newer independent pretrain"),
}
