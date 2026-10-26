"""Function-preserving residual-stream transforms for disguise experiments.

Conventions match HuggingFace CausalLM (weight shape out×in):
  residual row-vector h; linear y = h @ W.T.

Permutation (fold=False): h' = h @ P, P a permutation matrix.
  Reads:  W <- W @ P
  Writes: W <- P.T @ W
  RMSNorm weights are permuted (no fold).

Anisotropic scaling (fold=True first): h' = h @ S, S = diag(s).
  Fold each RMSNorm γ into following linears (SliceGPT/QuaRot), set γ=1.
  Reads:  W <- W @ S^{-1}
  Writes: W <- S @ W
  Embed:  E <- E @ S

Layer prune: drop selected decoder layers from model.model.layers in place.
"""
from __future__ import annotations

import torch
from torch import nn


def _untie_lm_head(model) -> None:
    m = model.model
    if model.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(m.embed_tokens.weight.detach().clone())
        model.config.tie_word_embeddings = False


@torch.no_grad()
def apply_linear_residual_transform(model, Q: torch.Tensor, *, fold: bool) -> None:
    """Apply h' = h @ Q with Q orthogonal (incl. permutation) or diagonal.

    fold=True: fold RMSNorm into following linears then apply Q (needed for
    non-permutation orthogonal maps and for anisotropic diagonal S).
    fold=False: only valid when Q is a permutation (RMSNorm reordered).
    """
    _untie_lm_head(model)
    m = model.model
    Q = Q.to(dtype=m.embed_tokens.weight.dtype, device=m.embed_tokens.weight.device)
    # For diagonal S we need S^{-1} on reads; for orthogonal/permutation Q^{-1}=Q.T
    if Q.shape[0] == Q.shape[1] and torch.allclose(
        Q @ Q.T, torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype), atol=1e-4
    ):
        Q_read = Q  # W <- W @ Q  (rotation / perm case, matches prior extract.py)
        Q_write = Q.T
        is_orthogonal = True
    else:
        # Diagonal (or general invertible): h' = h @ Q  => reads W @ Q^{-1}, writes Q @ W
        Q_inv = torch.linalg.inv(Q)
        Q_read = Q_inv
        Q_write = Q
        is_orthogonal = False

    if not fold and not is_orthogonal:
        raise ValueError("non-orthogonal Q requires fold=True (RMSNorm)")

    perm = None if fold else Q.argmax(0)

    def norm_and_reads(norm, linears):
        w = norm.weight.data
        if fold:
            for L in linears:
                L.weight.data = (L.weight.data * w[None, :]) @ Q_read
            norm.weight.data = torch.ones_like(w)
        else:
            for L in linears:
                L.weight.data = L.weight.data @ Q_read
            norm.weight.data = w[perm]

    m.embed_tokens.weight.data = m.embed_tokens.weight.data @ (
        Q if is_orthogonal else Q
    )
    # embed: E <- E @ Q always (h_embed' = token @ E @ Q)
    for layer in m.layers:
        a, f = layer.self_attn, layer.mlp
        norm_and_reads(layer.input_layernorm, [a.q_proj, a.k_proj, a.v_proj])
        norm_and_reads(layer.post_attention_layernorm, [f.gate_proj, f.up_proj])
        for L in (a.o_proj, f.down_proj):
            L.weight.data = Q_write @ L.weight.data
            if L.bias is not None:
                L.bias.data = L.bias.data @ (Q if is_orthogonal else Q)
    norm_and_reads(m.norm, [model.lm_head])


@torch.no_grad()
def make_permutation(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.eye(dim)[:, torch.randperm(dim, generator=g)]


@torch.no_grad()
def make_anisotropic_scale(dim: int, seed: int = 0, low: float = 0.25, high: float = 4.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    s = torch.empty(dim).uniform_(low, high, generator=g)
    return torch.diag(s)


@torch.no_grad()
def prune_layers(model, drop_indices: list[int]) -> list[int]:
    """Remove decoder layers at drop_indices (original indices). Returns kept indices."""
    layers = model.model.layers
    total = len(layers)
    drop = set(int(i) for i in drop_indices)
    if any(i < 0 or i >= total for i in drop):
        raise ValueError(f"drop indices must be in [0, {total})")
    if len(drop) >= total:
        raise ValueError("cannot drop all layers")
    kept = [i for i in range(total) if i not in drop]
    new_list = nn.ModuleList([layers[i] for i in kept])
    model.model.layers = new_list
    model.config.num_hidden_layers = len(kept)
    # Reindex attention layer_idx so KV-cache / hooks stay consistent
    for new_i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_i
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return kept
