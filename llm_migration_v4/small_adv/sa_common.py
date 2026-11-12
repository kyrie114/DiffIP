"""小模型优点验证（2026-11-12）的公共部分。

复用 llm_migration_v3 的协议与已采特征：受害模型 Qwen2.5-0.5B，衍生基底
Qwen2.5-0.5B-Instruct，5 个无关模型（SmolLM2-135M/360M、GPT-2、Pythia-160M、
Qwen2-0.5B）。扰动、池化、层取样完全沿用 v3 common（fp32、hook 取解码层输出、
token 平均池化、均匀 4 层）。本文件只新增在本机构造的衍生模型：
  dropK      对 Instruct 均匀删去 K 层（K=4 即 v3 已有的 instruct_prune）
  widthP     对 Instruct 的残差流按激活重要性只保留 P% 通道（结构化剪宽度）
  vocab_ext  在 Qwen 词表上新增 2000 个合并词元，子词均值初始化，再只训练词嵌入
  vocab_swap 整个换成 SmolLM2 的词表（49152），同名词元直接拷贝、其余子词均值
             初始化，再只训练词嵌入
"""
from __future__ import annotations
import csv, random, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn

SA = Path(__file__).resolve().parent
V3 = SA.parent.parent / "llm_migration_v3"
sys.path.insert(0, str(V3))
import common as c3  # noqa: E402  (v3 common: PROMPTS, SEED_SETS, perturb_batch, forward_pool, ...)
from transformers import AutoTokenizer  # noqa: E402

CKPT = SA / "ckpt"; CKPT.mkdir(exist_ok=True)
TQA_CSV = V3 / "reef_baseline" / "TruthfulQA.csv"
SMOL = "HuggingFaceTB/SmolLM2-135M"


def tqa_rows():
    rows = list(csv.DictReader(open(TQA_CSV, encoding="utf-8")))
    random.Random(2024).shuffle(rows)   # 与 v3 REEF 对照同一顺序：前 200 条给 REEF
    return [r["Question"].strip() + " " + r["Best Answer"].strip() for r in rows]


def corpus():
    """训练/校准语料：TruthfulQA 第 200 条以后（不碰 REEF 用的前 200 条）+ 本仓库与
    学习仓库里的中文 Markdown 文档（不含 32 条提示）。"""
    en = tqa_rows()[200:]
    zh = []
    for root in [SA.parent.parent, Path("/workspace/llm-security-study")]:
        for f in sorted(root.rglob("*.md")):
            if "small_adv" in str(f):
                continue
            t = f.read_text(encoding="utf-8", errors="ignore")
            for para in re.split(r"\n\s*\n", t):
                para = para.strip()
                if len(re.findall(r"[\u4e00-\u9fff]", para)) > 20:
                    zh.append(para[:600])
    prompts = set(c3.PROMPTS)
    zh = [z for z in zh if not any(p in z for p in prompts)]
    return en, zh


def drop_spec(total, k):
    return [round(i * (total - 1) / (k + 1)) for i in range(1, k + 1)]


def _slice_linear(lin, rows=None, cols=None):
    W = lin.weight.data
    if rows is not None: W = W[rows]
    if cols is not None: W = W[:, cols]
    b = lin.bias.data[rows] if (lin.bias is not None and rows is not None) else (lin.bias.data if lin.bias is not None else None)
    new = nn.Linear(W.shape[1], W.shape[0], bias=b is not None)
    new.weight.data = W.clone().contiguous()
    if b is not None: new.bias.data = b.clone()
    return new


@torch.no_grad()
def channel_importance(m, tok, texts):
    acc = None
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states
        a = torch.stack([h[0].abs().mean(0) for h in hs]).mean(0)
        acc = a if acc is None else acc + a
    return acc


@torch.no_grad()
def width_prune(m, keep_idx):
    keep = torch.as_tensor(sorted(keep_idx.tolist()))
    E = m.model.embed_tokens
    newE = nn.Embedding(E.weight.shape[0], len(keep)); newE.weight.data = E.weight.data[:, keep].clone()
    m.model.embed_tokens = newE
    for L in m.model.layers:
        a, f = L.self_attn, L.mlp
        a.q_proj = _slice_linear(a.q_proj, cols=keep); a.k_proj = _slice_linear(a.k_proj, cols=keep)
        a.v_proj = _slice_linear(a.v_proj, cols=keep); a.o_proj = _slice_linear(a.o_proj, rows=keep)
        f.gate_proj = _slice_linear(f.gate_proj, cols=keep); f.up_proj = _slice_linear(f.up_proj, cols=keep)
        f.down_proj = _slice_linear(f.down_proj, rows=keep)
        for nm in ("input_layernorm", "post_attention_layernorm"):
            ln = getattr(L, nm); ln.weight = nn.Parameter(ln.weight.data[keep].clone())
    m.model.norm.weight = nn.Parameter(m.model.norm.weight.data[keep].clone())
    m.lm_head = nn.Linear(len(keep), newE.weight.shape[0], bias=False); m.lm_head.weight = newE.weight
    m.config.hidden_size = len(keep)
    return m


@torch.no_grad()
def set_vocab(m, W):
    """W: [V_new, D] 新词嵌入；lm_head 与词嵌入共享（Qwen2.5-0.5B 本来就共享）。"""
    newE = nn.Embedding(W.shape[0], W.shape[1]); newE.weight.data = W.clone()
    m.model.embed_tokens = newE
    m.lm_head = nn.Linear(W.shape[1], W.shape[0], bias=False); m.lm_head.weight = newE.weight
    m.config.vocab_size = W.shape[0]
    return m


def merge_tokens(tok, texts, n):
    cnt = Counter()
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids
        for a, b in zip(ids, ids[1:]):
            cnt[(a, b)] += 1
    out, seen = [], set(tok.get_vocab())
    for (a, b), _ in cnt.most_common():
        s = tok.decode([a, b])
        if "\n" in s or len(s.strip()) < 2 or "\ufffd" in s or s in seen:
            continue
        seen.add(s); out.append((s, a, b))
        if len(out) >= n:
            break
    return out


def train_embeddings(m, tok, texts, steps, lr=1e-4, seq=128, bs=4, seed=0, log=print, which="embed"):
    """which="embed"：只训练（共享的）词嵌入，其余冻结，模仿中文 LLaMA 扩词表后的第一阶段。
    which="body"：冻结词嵌入，训练其余全部参数（剪宽度后的短暂恢复训练）。"""
    W = m.model.embed_tokens.weight
    for p in m.parameters(): p.requires_grad_(which == "body")
    W.requires_grad_(which == "embed")
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False).input_ids + [tok.eos_token_id or 0])
    ids = torch.tensor(ids)
    g = torch.Generator().manual_seed(seed)
    m.train()
    for s in range(steps):
        st = torch.randint(0, len(ids) - seq - 1, (bs,), generator=g)
        x = torch.stack([ids[i:i + seq] for i in st])
        loss = m(input_ids=x, labels=x, use_cache=False).loss
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 25 == 0 or s == steps - 1:
            log(f"    step {s} loss {loss.item():.3f}")
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


def build(name):
    """Returns (model, tok, embed_fn) — 与 v3 build_model 接口一致（不含置换项）。"""
    if name in ("victim", "instruct", "instruct_wperm", "instruct_prune") or name in c3.NEGS:
        m, t, emb, _ = c3.build_model(name)
        return m, t, emb
    if name.startswith("drop"):
        m, t = c3.load(c3.INSTRUCT)
        c3.prune_layers(m, drop_spec(m.config.num_hidden_layers, int(name[4:])))
        E = m.get_input_embeddings(); return m, t, (lambda ids: E(ids))
    d = CKPT / name
    if not d.exists():
        raise SystemExit(f"run sa_build.py {name} first")
    r = torch.load(d / "recipe.pt")
    m, _ = c3.load(c3.INSTRUCT)
    if name.startswith("width"):
        width_prune(m, r["keep"])
        if "body" in r:
            m.load_state_dict(r["body"], strict=False)
    else:
        set_vocab(m, r["embed"])
    m.eval()
    t = AutoTokenizer.from_pretrained(d / "tok")
    E = m.get_input_embeddings(); return m, t, (lambda ids: E(ids))


@torch.no_grad()
def ppl(m, tok, texts):
    nll, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        if ids.shape[1] < 2: continue
        out = m(input_ids=ids, labels=ids, use_cache=False)
        nll += out.loss.item() * (ids.shape[1] - 1); n += ids.shape[1] - 1
    return float(np.exp(nll / n))
