"""DiffIP-faithful scoring: PCA-20 + per-layer ALS (QSX-h), NO DP path search.

Primary metric = mean over layers of fit_step_alignment(...).distance
  which is (1/N) || Q (S ⊙ X_c) - Y_c ||_F^2   (official reversion.py)

similarity_als = exp( - sum_l d_l / L )   # matches DiffIPMetric when DP
takes the strict diagonal on an L×L local-distance matrix (no skip).

Also reports DiffIPMetric.compare() for reference (still runs DP internally;
with equal L it usually hugs the diagonal — documented, not used as primary).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from diffip.metric import DiffIPMetric
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment


@dataclass
class ALSOnlyResult:
    distance_mean: float          # mean over layers of per-step mean-L2
    distance_sum: float           # sum over layers (DiffIP path cost if diagonal)
    similarity: float             # exp(-distance_sum / L)
    per_layer_distance: list[float]
    n_layers: int
    n_components: int
    converged_all: bool


def als_only_compare(
    suspect: npt.ArrayLike,
    victim: npt.ArrayLike,
    *,
    n_components: int = 20,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> ALSOnlyResult:
    """1:1 layer pairing + ALS. Requires equal number of layers (DP unused)."""
    projected = project_sequences(suspect, victim, n_components)
    if projected.suspect.shape[0] != projected.victim.shape[0]:
        raise ValueError(
            "als_only_compare requires equal layer counts; "
            f"got {projected.suspect.shape[0]} vs {projected.victim.shape[0]}. "
            "For unequal depth use DiffIPMetric (DP) or matched_subset_layers()."
        )
    L = projected.suspect.shape[0]
    per_layer: list[float] = []
    converged_all = True
    for i in range(L):
        al = fit_step_alignment(
            projected.suspect[i],
            projected.victim[i],
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        per_layer.append(float(al.distance))
        converged_all = converged_all and bool(al.converged)
    distance_sum = float(sum(per_layer))
    distance_mean = distance_sum / L
    similarity = math.exp(-distance_sum / L)
    return ALSOnlyResult(
        distance_mean=distance_mean,
        distance_sum=distance_sum,
        similarity=similarity,
        per_layer_distance=per_layer,
        n_layers=L,
        n_components=projected.n_components,
        converged_all=converged_all,
    )


def matched_subset_layers(
    reps: np.ndarray, n_keep: int
) -> np.ndarray:
    """Evenly subsample layer axis to n_keep (for prune matched-subset ALS)."""
    L = reps.shape[0]
    if n_keep > L:
        raise ValueError(f"n_keep={n_keep} > L={L}")
    if n_keep == L:
        return reps
    idx = [round(i * (L - 1) / (n_keep - 1)) for i in range(n_keep)]
    return reps[idx]


def score_pair(
    suspect: np.ndarray,
    victim: np.ndarray,
    *,
    n_components: int = 20,
    also_full_metric: bool = True,
) -> dict:
    """Return ALS-only primary scores (+ optional DiffIPMetric reference)."""
    # If unequal L, match by even subsample of the longer one
    Ls, Lv = suspect.shape[0], victim.shape[0]
    note = None
    if Ls != Lv:
        n = min(Ls, Lv)
        suspect = matched_subset_layers(suspect, n)
        victim = matched_subset_layers(victim, n)
        note = (
            f"unequal depth {Ls} vs {Lv}: matched-subset ALS on {n} even layers "
            "(DP not used; this is NOT DiffIP's DP alignment)"
        )
    als = als_only_compare(suspect, victim, n_components=n_components)
    out = {
        "primary": "als_only_mean_L2",
        "als": asdict(als),
        "note": note,
    }
    if also_full_metric:
        # DiffIPMetric always runs DP; with equal L after match it is reference only
        ref = DiffIPMetric(n_components=n_components).compare(suspect, victim)
        out["diffip_metric_reference"] = {
            "similarity": float(ref.similarity),
            "distance": float(ref.distance),
            "warning": "includes DP; not the primary number for this experiment",
        }
    return out
