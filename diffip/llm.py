"""LLM representation collector bridging transformers models to DiffIP.

The collector treats the model's layers as the temporal axis (analogous to
denoising steps in diffusion models) and gathers one mean-pooled hidden-state
vector per layer for each of ``N`` perturbed forward passes. The output layout
``[L, N, D]`` matches the ``[T, N, D]`` sequences expected by
:class:`diffip.DiffIPMetric`.

Requires optional dependencies: ``torch`` and ``transformers``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

try:  # optional dependencies, imported lazily so the core package stays light
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
except ImportError as _error:  # pragma: no cover - exercised only without extras
    torch = None  # type: ignore[assignment]
    PreTrainedModel = None  # type: ignore[assignment,misc]
    PreTrainedTokenizerBase = None  # type: ignore[assignment,misc]
    _IMPORT_ERROR = _error


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "diffip.llm requires torch and transformers; "
            "install them with: python -m pip install torch transformers"
        ) from _IMPORT_ERROR


@dataclass(frozen=True)
class CollectedRepresentations:
    """Representation sequence collected from one model."""

    representations: npt.NDArray[np.float64]  # [L, N, D]
    layer_indices: tuple[int, ...]
    prompt: str


class LLMRepresentationCollector:
    """Collect per-layer representations from a causal LM under input perturbations.

    Parameters
    ----------
    model, tokenizer:
        A Hugging Face causal language model and its tokenizer. The model is
        put in eval mode; gradients are disabled during collection.
    n_samples:
        Number of perturbed forward passes (analogous to random seeds in the
        diffusion setting). Must be >= 2 so per-sample statistics exist.
    perturbation_scale:
        Standard deviation of the Gaussian noise added to the input
        embeddings for each sample beyond the first.
    layers:
        Optional explicit layer indices whose hidden states are collected
        (0-based, counting only decoder layers). Defaults to every layer.
    device:
        Optional device string, e.g. ``"cuda"``. Defaults to the model's own
        device.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        n_samples: int = 8,
        perturbation_scale: float = 1e-2,
        layers: Sequence[int] | None = None,
        device: str | None = None,
        sample_mode: str = "perturbation",
        prompt_pool: Sequence[str] | None = None,
        n_perturbations: int = 2,
        seed: int | None = None,
    ) -> None:
        _require_torch()
        if sample_mode not in ("perturbation", "prompts", "hybrid"):
            raise ValueError(
                "sample_mode must be 'perturbation', 'prompts' or 'hybrid'"
            )
        if sample_mode == "perturbation":
            if n_samples < 2:
                raise ValueError("n_samples must be at least 2")
            if perturbation_scale <= 0:
                raise ValueError("perturbation_scale must be positive")
        elif sample_mode == "prompts":
            if prompt_pool is None or len(prompt_pool) < 2:
                raise ValueError(
                    "sample_mode='prompts' requires a prompt_pool with at least 2 prompts"
                )
            n_samples = len(prompt_pool)
        else:  # hybrid
            if prompt_pool is None or len(prompt_pool) < 2:
                raise ValueError(
                    "sample_mode='hybrid' requires a prompt_pool with at least 2 prompts"
                )
            if n_perturbations < 1:
                raise ValueError("n_perturbations must be at least 1")
            if perturbation_scale <= 0:
                raise ValueError("perturbation_scale must be positive")
            n_samples = len(prompt_pool) * n_perturbations
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.perturbation_scale = perturbation_scale
        self.sample_mode = sample_mode
        self.prompt_pool = tuple(prompt_pool) if prompt_pool is not None else None
        self.n_perturbations = n_perturbations
        self.seed = seed
        self.device = device or next(model.parameters()).device
        self.model.eval()

        self.layer_indices = self._resolve_layers(layers)
        self._hidden: dict[int, "torch.Tensor"] = {}
        self._hooks: list = []

    def _resolve_layers(self, layers: Sequence[int] | None) -> tuple[int, ...]:
        total = self.model.config.num_hidden_layers
        if layers is None:
            return tuple(range(total))
        resolved = tuple(int(index) for index in layers)
        if not resolved:
            raise ValueError("layers must be non-empty when provided")
        if any(index < 0 or index >= total for index in resolved):
            raise ValueError(f"layer indices must be within [0, {total})")
        return resolved

    # ------------------------------------------------------------------
    def _install_hooks(self) -> None:
        decoder_layers = self.model.model.layers

        def make_hook(layer_index: int):
            def hook(_module, _inputs, output):
                # decoder layers return either a tensor or a tuple whose first
                # element is the hidden state
                hidden = output[0] if isinstance(output, tuple) else output
                self._hidden[layer_index] = hidden.detach()

            return hook

        for index in self.layer_indices:
            self._hooks.append(
                decoder_layers[index].register_forward_hook(make_hook(index))
            )

    def _remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    def _build_samples(self, prompt: str | None):
        """Return one (inputs_embeds, attention_mask) pair per fingerprint sample."""
        embed_layer = self.model.get_input_embeddings()

        def tokenized(text: str):
            encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
            return embed_layer(encoded["input_ids"]), encoded["attention_mask"]

        def perturbed_copies(inputs_embeds, attention_mask, count: int):
            """One clean copy plus (count - 1) Gaussian-perturbed copies."""
            generator = (
                torch.Generator().manual_seed(self.seed)
                if self.seed is not None
                else None
            )
            samples = []
            for sample_index in range(count):
                if sample_index == 0:
                    samples.append((inputs_embeds, attention_mask))
                    continue
                if generator is not None:
                    noise = torch.randn(
                        inputs_embeds.shape,
                        generator=generator,
                        dtype=inputs_embeds.dtype,
                    ).to(inputs_embeds.device)
                else:
                    noise = torch.randn_like(inputs_embeds)
                samples.append(
                    (inputs_embeds + self.perturbation_scale * noise, attention_mask)
                )
            return samples

        if self.sample_mode == "prompts":
            return [tokenized(text) for text in self.prompt_pool]  # type: ignore[union-attr]

        if self.sample_mode == "hybrid":
            samples = []
            for text in self.prompt_pool:  # type: ignore[union-attr]
                inputs_embeds, attention_mask = tokenized(text)
                samples.extend(
                    perturbed_copies(
                        inputs_embeds, attention_mask, self.n_perturbations
                    )
                )
            return samples

        if prompt is None:
            raise ValueError("collect(prompt) requires a prompt in perturbation mode")
        inputs_embeds, attention_mask = tokenized(prompt)
        return perturbed_copies(inputs_embeds, attention_mask, self.n_samples)

    # ------------------------------------------------------------------
    def collect(self, prompt: str | None = None) -> CollectedRepresentations:
        """Run ``n_samples`` forwards and pool per-layer states.

        In ``perturbation`` mode sample 0 uses the clean input and every
        subsequent sample adds iid Gaussian noise to the input embeddings.
        In ``prompts`` mode each sample is one forward pass on a distinct
        prompt from the configured pool.
        """
        _require_torch()
        samples = self._build_samples(prompt)

        layer_vectors: list[list[np.ndarray]] = [
            [] for _ in self.layer_indices
        ]
        with torch.no_grad():
            for inputs_embeds, attention_mask in samples:
                self._hidden.clear()
                self._install_hooks()
                try:
                    self.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                finally:
                    self._remove_hooks()

                mask = attention_mask.unsqueeze(-1).to(
                    self._hidden[self.layer_indices[0]].dtype
                )
                for position, layer_index in enumerate(self.layer_indices):
                    hidden = self._hidden[layer_index]
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                    layer_vectors[position].append(
                        pooled.squeeze(0).to(torch.float64).cpu().numpy()
                    )

        representations = np.stack(
            [np.stack(vectors, axis=0) for vectors in layer_vectors], axis=0
        )
        label = (
            f"<{len(self.prompt_pool)} prompts>"
            if self.prompt_pool is not None
            else (prompt or "")
        )
        return CollectedRepresentations(
            representations=representations,
            layer_indices=self.layer_indices,
            prompt=label,
        )
