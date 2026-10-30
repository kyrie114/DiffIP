"""Dynamic diagnosis features. Same perturbation protocol as ../lit_protocol/collect.py
(32 prompts, SEED_SETS, N=128, sigma 2e-3, mean pooling, fp32), plus all-layer features on
seed set 0 for the per-layer sim curve. Variants:
  victim / instruct / coder_base / coder_inst / r1_draft / qwen2_inst : unmodified
  r1_draft_victok   : R1-DRAFT weights, victim tokenizer (no <|begin_of_sentence|> BOS)
  victim_bos        : victim weights, R1-DRAFT tokenizer (BOS 151646 prepended)  [control]
  coder_base_vicemb / coder_inst_vicemb : Coder weights with victim embed_tokens swapped in
Writes feats/<variant>.npz: pert [32,3,4,N,D] (L4), all0 [32,total,N,D] float16."""
import sys, time
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H.parent))
from common import *
H = Path(__file__).resolve().parent  # common exports its own H
HF = {"victim": VICTIM, "instruct": INSTRUCT, "coder_base": "Qwen/Qwen2.5-Coder-0.5B",
      "coder_inst": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "r1_draft": "alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B",
      "qwen2_inst": "Qwen/Qwen2-0.5B-Instruct"}
VAR = {k: (v, None, None) for k, v in HF.items()}
VAR["r1_draft_victok"] = (HF["r1_draft"], VICTIM, None)
VAR["victim_bos"] = (VICTIM, HF["r1_draft"], None)
VAR["coder_base_vicemb"] = (HF["coder_base"], None, "emb")
VAR["coder_inst_vicemb"] = (HF["coder_inst"], None, "emb")
OUT = H / "feats"; OUT.mkdir(exist_ok=True)
for key in (sys.argv[1:] or VAR):
    f = OUT / f"{key}.npz"
    if f.exists(): print("skip", key); continue
    mid, tokid, swap = VAR[key]; t0 = time.time()
    m, tok = load(mid)
    if tokid: tok = AutoTokenizer.from_pretrained(tokid)
    if swap == "emb":
        from huggingface_hub import snapshot_download; from safetensors import safe_open
        d = Path(snapshot_download(VICTIM, allow_patterns=["*.safetensors"]))
        h = safe_open(str(next(d.glob("*.safetensors"))), "pt")  # victim is single-file
        with torch.no_grad(): m.get_input_embeddings().weight.copy_(h.get_tensor("model.embed_tokens.weight").float())
    E = m.get_input_embeddings(); total = len(decoder_layers(m)); L4 = even_layers(total, 4); ALL = list(range(total))
    pert, all0 = [], []
    for p in PROMPTS:
        base = E(tok(p, return_tensors="pt")["input_ids"]).detach(); ps = []
        for j, s in enumerate(SEED_SETS):
            if j == 0:
                a = forward_pool(m, perturb_batch(base, N_MAX, s), ALL, chunk=32); all0.append(a.astype(np.float16)); ps.append(a[L4])
            else: ps.append(forward_pool(m, perturb_batch(base, N_MAX, s), L4, chunk=32))
        pert.append(np.stack(ps))
    np.savez(f, pert=np.stack(pert).astype(np.float32), all0=np.stack(all0), layers4=np.array(L4), total=total)
    print(key, f"{time.time()-t0:.0f}s", flush=True); del m; gc.collect()
