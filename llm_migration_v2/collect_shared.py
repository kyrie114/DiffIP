"""Collect [L,N,D] reps for a (possibly transformed) model on SHARED inputs_embeds.

For weight-space disguises (permutation), each model must see the *same*
embedding cloud. If the suspect's embed_tokens were transformed (E@P), calling
its own embed_layer + iid noise breaks the global linear relation across the
N samples. Build noise once from a reference tokenizer/model, then reuse.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def build_perturbed_embeds(
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    n_samples: int,
    perturbation_scale: float,
    seed: int,
    device=None,
):
    device = device or next(ref_model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    base = ref_model.get_input_embeddings()(encoded["input_ids"])
    mask = encoded["attention_mask"]
    g = torch.Generator().manual_seed(seed)
    samples = []
    for i in range(n_samples):
        if i == 0:
            samples.append((base.clone(), mask))
            continue
        noise = torch.randn(base.shape, generator=g, dtype=base.dtype, device="cpu").to(device)
        samples.append((base + perturbation_scale * noise, mask))
    return samples


@torch.no_grad()
def collect_on_embeds(
    model: PreTrainedModel,
    samples,
    layers: Sequence[int],
) -> np.ndarray:
    """Hook decoder layers; mean-pool; return [L,N,D]."""
    model.eval()
    layer_indices = tuple(int(i) for i in layers)
    hidden: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(idx):
        def hook(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden[idx] = h.detach()
        return hook

    decoder = model.model.layers
    layer_vectors: list[list[np.ndarray]] = [[] for _ in layer_indices]
    for inputs_embeds, attention_mask in samples:
        hidden.clear()
        hooks = [decoder[i].register_forward_hook(make_hook(i)) for i in layer_indices]
        try:
            model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        mask = attention_mask.unsqueeze(-1).to(hidden[layer_indices[0]].dtype)
        for pos, idx in enumerate(layer_indices):
            h = hidden[idx]
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)
            layer_vectors[pos].append(pooled.squeeze(0).to(torch.float64).cpu().numpy())
    return np.stack([np.stack(v, axis=0) for v in layer_vectors], axis=0)
