"""FingerInv (CVPR25) Table-2 attack completion for the LLM v3 panel.
Attacks are stacked on the fine-tuned suspect (Qwen2.5-0.5B-Instruct), exactly like the
v3 `instruct_prune` row; victim = Qwen2.5-0.5B (fp32, existing feats/pert_victim.npz).
Protocol identical to ../collect_perturb.py: same 32 prompts, SEED_SETS, N_MAX=128,
SCALE=2e-3, 4 even layers, mean pooling. Only deviation: dp_all (all-layer, for DP-on)
is not stored. Perturbation noise is generated in fp32 (as in v3), then cast to the
attack dtype before the forward.
Attacks:
  q_w8sim / q_w4sim (extra, SIMULATED): per-output-channel symmetric round-to-nearest
      fake-quant of every decoder Linear weight, dequantized back to fp32 for compute.
  q_fp16 / q_bf16 : weights AND forward in that dtype (CPU kernels exist in torch>=2).
  q_int8dyn (extra): torch.ao dynamic int8 quantization of all nn.Linear, fp32 activations.
  lprune_{5,10,15,30,50}: same method as v3 prune (drop whole decoder layers, evenly
      spaced via the prune_spec formula), n_drop = round(frac*24) (min 1).
  mprune_{...} (extra, FingerInv-style): global-per-tensor L1 magnitude unstructured
      pruning of every Linear weight in decoder layers at that rate.
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from common import *
OUT = Path(__file__).resolve().parent / "feats"; OUT.mkdir(exist_ok=True)
FR = {5: .05, 10: .10, 15: .15, 30: .30, 50: .50}

def layer_drop(total, frac):
    n = max(1, round(frac * total))
    return [round(i * (total - 1) / (n + 1)) for i in range(1, n + 1)]

def build(name):
    m, t = load(INSTRUCT); dt = torch.float32
    if name == "q_fp16": dt = torch.float16
    elif name == "q_bf16": dt = torch.bfloat16
    elif name == "q_int8dyn":
        m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    elif name in ("q_w8sim", "q_w4sim"):  # SIMULATED fake-quant: per-output-channel symmetric RTN, fp32 compute
        bits = 8 if name == "q_w8sim" else 4; qmax = 2 ** (bits - 1) - 1
        with torch.no_grad():
            for mod in m.model.layers.modules():
                if isinstance(mod, torch.nn.Linear):
                    w = mod.weight; sc = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
                    w.copy_((w / sc).round().clamp(-qmax - 1, qmax) * sc)
    elif name.startswith("lprune_"):
        spec = layer_drop(m.config.num_hidden_layers, FR[int(name.split("_")[1])])
        print(name, "drop", spec, flush=True); prune_layers(m, spec)
    elif name.startswith("mprune_"):
        r = FR[int(name.split("_")[1])]
        with torch.no_grad():
            for mod in m.model.layers.modules():
                if isinstance(mod, torch.nn.Linear):
                    w = mod.weight; k = int(r * w.numel())
                    if k: thr = w.abs().flatten().kthvalue(k).values; w[w.abs() <= thr] = 0
    m = m.to(dt).eval()
    return m, t, dt

names = sys.argv[1:]
for name in names:
    f = OUT / f"pert_{name}.npz"
    if f.exists(): print("skip", name); continue
    t0 = time.time(); m, tok, dt = build(name)
    E = m.get_input_embeddings() if name != "q_int8dyn" else m.model.embed_tokens
    total = len(decoder_layers(m)); L4 = even_layers(total, 4)
    pert = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt")["input_ids"]
        base = E(ids).detach().to(torch.float32)
        sets = [forward_pool(m, perturb_batch(base, N_MAX, s).to(dt), L4, chunk=32) for s in SEED_SETS]
        pert.append(np.stack(sets))
    np.savez(f, pert=np.stack(pert).astype(np.float32), layers4=np.array(L4), total=total)
    print(name, "L4", L4, f"{time.time()-t0:.0f}s", flush=True)
    del m; gc.collect()
