"""The only way to turn llm_migration_v4 score JSON into the kept lineage decision.

Callers pass a directory of existing ``main_*.json`` rows. This module does not
rescore, does not call ALS, and does not write ``scores/`` or ``scores_mid2/``.
Callers never receive the similarity table and never index it.

``decision().passed`` is the only pass/fail. It is experiment D's aligned
margin at even4, N=128, d=8, both anchor directions, and only for the strong
and SFT groups. Win counts are descriptive and are not a pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


__all__ = [
    "BothDirections",
    "DecisionSetting",
    "DirectionResult",
    "LineageDataError",
    "LineageDecision",
    "LineagePanel",
    "Miss",
    "ModelRole",
    "PromptWinCount",
    "SeedRates",
]

_TOKEN = object()
_N_PROMPTS = 32
_DECISION_N = 128
_DECISION_DIMS = 8
_WIN_NS = (20, 128)
_WIN_DIMS = (8, 20)
_CALIBRATE_SEED = 0
_HELDOUT_SEEDS = (1, 2)
_TPR_MIN = 0.90
_FPR_MAX = 0.10
# even4 fine_tune at N=128 d=8 is 0.363. mid2 is 0.249.
# A folder named scores on the mid2 side of this gap is not the even4 corpus.
_EVEN4_FT_MEAN_FLOOR = 0.30
_SETTING = ("even4", _DECISION_N, _DECISION_DIMS)


class LineageDataError(Exception):
    """The score directory is not a complete main-line corpus.

    Raised at load for an unknown directory name, a roster/filename mismatch,
    a malformed row, a non-finite similarity, or a missing decision cell.
    """


class ModelRole(Enum):
    """Role assigned by the panel. Callers do not classify file names."""

    VICTIM = "victim"
    STRONG = "strong"
    SFT = "sft"
    HARD = "hard"


class _Cohort(Enum):
    VICTIM = "victim"
    STRONG = "strong"
    SFT = "sft"
    HARD = "hard"
    CROSS = "cross"


# One membership list. Exact stems. Never a neg_ prefix test.
# neg_pruned_* are strong positives. The prefix is historical.
_ROSTER: tuple[tuple[str, _Cohort], ...] = (
    ("victim", _Cohort.VICTIM),
    ("neg_pruned_sparsegpt", _Cohort.STRONG),
    ("neg_pruned_wanda", _Cohort.STRONG),
    ("neg_pruned_gblm", _Cohort.STRONG),
    ("fine_tune", _Cohort.SFT),
    ("weight_perm", _Cohort.SFT),
    ("rep_perm", _Cohort.SFT),
    ("scale", _Cohort.SFT),
    ("prune_dpoff", _Cohort.SFT),
    ("derivative_cn", _Cohort.HARD),
    ("neg_qwen2_5_7b", _Cohort.CROSS),
    ("neg_llama3_8b", _Cohort.CROSS),
)


def _closed_roster() -> dict[str, _Cohort]:
    stems = [stem for stem, _ in _ROSTER]
    if len(stems) != len(set(stems)):
        raise LineageDataError("roster stems are not unique")
    counts = {cohort: 0 for cohort in _Cohort}
    for _, cohort in _ROSTER:
        counts[cohort] += 1
    expected = {
        _Cohort.VICTIM: 1,
        _Cohort.STRONG: 3,
        _Cohort.SFT: 5,
        _Cohort.HARD: 1,
        _Cohort.CROSS: 2,
    }
    if counts != expected:
        raise LineageDataError(f"roster counts {counts} != {expected}")
    return dict(_ROSTER)


_ROLE_OF = _closed_roster()
_BY_COHORT: dict[_Cohort, tuple[str, ...]] = {
    cohort: tuple(stem for stem, got in _ROSTER if got is cohort) for cohort in _Cohort
}
_CROSS = _BY_COHORT[_Cohort.CROSS]
_WIN_MODELS = (
    _BY_COHORT[_Cohort.VICTIM]
    + _BY_COHORT[_Cohort.SFT]
    + _BY_COHORT[_Cohort.HARD]
    + _BY_COHORT[_Cohort.STRONG]
)
_PUBLIC_ROLE = {
    _Cohort.VICTIM: ModelRole.VICTIM,
    _Cohort.STRONG: ModelRole.STRONG,
    _Cohort.SFT: ModelRole.SFT,
    _Cohort.HARD: ModelRole.HARD,
}


@dataclass(frozen=True)
class DecisionSetting:
    """The only setting that can pass. Callers do not construct one to ask."""

    layers: str
    n: int
    dims: int
    calibrate_seed: int
    heldout_seeds: tuple[int, int]


@dataclass(frozen=True)
class SeedRates:
    """Held-out rates for one seed. Seed 0 is not represented.

    ``deploy_fpr`` is the false-positive rate of this direction's test model.
    ``hard_tpr`` does not gate the pass.
    """

    seed: int
    strong_tpr: float
    sft_tpr: float
    deploy_fpr: float
    hard_tpr: float


@dataclass(frozen=True)
class Miss:
    """A held-out strong or SFT cell whose margin is below the threshold.

    Identity of a shortfall, not a second verdict. ``seed`` is 1 or 2.
    """

    model: str
    prompt: int
    seed: int
    margin: float


@dataclass(frozen=True)
class DirectionResult:
    """One assignment of anchor and test. Not itself a pass/fail.

    ``threshold`` is the max seed-0 margin of ``test``. A positive on this
    direction is ``sim(model) - sim(anchor) >= threshold`` at the same
    prompt and seed. There is no method that accepts these rates back.
    """

    anchor: str
    test: str
    threshold: float
    heldout: tuple[SeedRates, SeedRates]
    strong_misses: tuple[Miss, ...]
    sft_misses: tuple[Miss, ...]


@dataclass(frozen=True)
class BothDirections:
    """Both cross-family assignments. One direction is not the decision.

    ``anchor_qwen.test`` is the Llama-3 key. ``anchor_llama.test`` is the
    Qwen key. ``setting`` is even4, N=128, dims=8, calibrate seed 0,
    held-out seeds (1, 2).
    """

    setting: DecisionSetting
    anchor_qwen: DirectionResult
    anchor_llama: DirectionResult


@dataclass(frozen=True)
class LineageDecision:
    """Panel pass/fail owned by ``LineagePanel``.

    ``passed`` is true only when ``directions`` is present and both
    directions meet the gate. ``directions`` is None when this directory
    is not the even4 corpus; ``passed`` is then false and no rates exist
    to quote. False with ``directions`` present means the even4 gate failed.
    """

    passed: bool
    directions: BothDirections | None


@dataclass(frozen=True)
class PromptWinCount:
    """Prompts strictly above every cross-family model. Not a pass.

    Seed is 0. ``n`` is 20 or 128. ``dims`` is 8 or 20. ``of`` is 32.
    ``role`` is never a cross-family model. No field here gates
    ``LineageDecision.passed``.
    """

    model: str
    role: ModelRole
    n: int
    dims: int
    won: int
    of: int


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LineageDataError(f"similarity is not a number: {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise LineageDataError("similarity is not finite")
    return number


def _load_rows(directory: Path) -> dict[tuple[str, int, int, int, int], float]:
    stems = {item.stem[len("main_"):] for item in directory.glob("main_*.json")}
    if stems != set(_ROLE_OF):
        missing = sorted(set(_ROLE_OF) - stems)
        extra = sorted(stems - set(_ROLE_OF))
        raise LineageDataError(f"roster mismatch missing={missing} extra={extra}")
    needed = {
        (n, dims, prompt, seed)
        for n in _WIN_NS
        for dims in _WIN_DIMS
        for prompt in range(_N_PROMPTS)
        for seed in (_CALIBRATE_SEED, *_HELDOUT_SEEDS)
    }
    rows: dict[tuple[str, int, int, int, int], float] = {}
    for stem in _ROLE_OF:
        payload = json.loads((directory / f"main_{stem}.json").read_text())
        if not isinstance(payload, list):
            raise LineageDataError(f"main_{stem}.json is not a list")
        seen: set[tuple[int, int, int, int]] = set()
        for row in payload:
            if not isinstance(row, dict):
                raise LineageDataError(f"main_{stem}.json has a non-object row")
            if row.get("pair") != stem:
                raise LineageDataError(f"main_{stem}.json pair is {row.get('pair')!r}")
            key = (int(row["prompt"]), int(row["seed_set"]), int(row["N"]), int(row["dims"]))
            if key not in {(p, s, n, d) for n, d, p, s in needed}:
                continue
            if key in seen:
                raise LineageDataError(f"duplicate cell in main_{stem}.json {key}")
            seen.add(key)
            rows[(stem, key[0], key[1], key[2], key[3])] = _finite(row["similarity"])
        if seen != {(p, s, n, d) for n, d, p, s in needed}:
            raise LineageDataError(f"main_{stem}.json is missing decision or win cells")
    return rows


class LineagePanel:
    """Owns the loaded main-line scores and the lineage decision.

    No public attribute returns a row, a similarity, or a mapping.
    ``from_score_dir`` is the only constructor.

    One membership list, inside this module, not exported. Exact names,
    never a ``neg_`` prefix test.
    """

    def __init__(self, rows: dict[tuple[str, int, int, int, int], float], *, allow_pass: bool, token: object) -> None:
        if token is not _TOKEN:
            raise LineageDataError("use LineagePanel.from_score_dir")
        self._rows = rows
        self._allow_pass = allow_pass

    @classmethod
    def from_score_dir(cls, path: str | Path) -> LineagePanel:
        """Read ``main_*.json`` and keep the rows inside the new panel.

        Ignores ``distill_*.json`` and ``summary_*.json``. Basename ``scores``
        is the even4 corpus only when the fine_tune mean at N=128, d=8 is
        at least 0.30. A mid2 tree renamed ``scores`` raises. Basename
        ``scores_mid2`` loads for win counts but cannot pass. Any other
        basename raises ``LineageDataError``.

        The ``main_*.json`` stems must equal the membership list exactly, or
        this raises. Each required cell (N in {20, 128}, dims in {8, 20},
        prompts 0..31, seeds 0..2) must exist once, ``pair`` must match the
        file stem, and ``similarity`` must be finite. Does not write the
        directory. Calling this twice on the same files yields equal decisions.
        """
        directory = Path(path)
        if not directory.is_dir():
            raise LineageDataError(f"not a directory: {directory}")
        name = directory.name
        if name == "scores":
            allow_pass = True
        elif name == "scores_mid2":
            allow_pass = False
        else:
            raise LineageDataError(
                f"directory basename {name!r} is not scores or scores_mid2"
            )
        rows = _load_rows(directory)
        if allow_pass:
            total = sum(
                rows[("fine_tune", prompt, seed, _DECISION_N, _DECISION_DIMS)]
                for prompt in range(_N_PROMPTS)
                for seed in range(3)
            )
            mean = total / (_N_PROMPTS * 3)
            if mean < _EVEN4_FT_MEAN_FLOOR:
                raise LineageDataError(
                    f"directory is named scores but fine_tune N=128 d=8 mean "
                    f"is {mean:.3f}, below {_EVEN4_FT_MEAN_FLOOR:.2f}"
                )
        return cls(rows, allow_pass=allow_pass, token=_TOKEN)

    def _sim(self, model: str, prompt: int, seed: int, n: int, dims: int) -> float:
        return self._rows[(model, prompt, seed, n, dims)]

    def _margin(self, model: str, anchor: str, prompt: int, seed: int) -> float:
        return (
            self._sim(model, prompt, seed, _DECISION_N, _DECISION_DIMS)
            - self._sim(anchor, prompt, seed, _DECISION_N, _DECISION_DIMS)
        )

    def _direction(self, anchor: str, test: str) -> DirectionResult:
        threshold = max(
            self._margin(test, anchor, prompt, _CALIBRATE_SEED) for prompt in range(_N_PROMPTS)
        )
        heldout: list[SeedRates] = []
        strong_misses: list[Miss] = []
        sft_misses: list[Miss] = []
        groups = {
            _Cohort.STRONG: _BY_COHORT[_Cohort.STRONG],
            _Cohort.SFT: _BY_COHORT[_Cohort.SFT],
            _Cohort.HARD: _BY_COHORT[_Cohort.HARD],
        }
        for seed in _HELDOUT_SEEDS:
            rates: dict[_Cohort, tuple[int, int]] = {}
            for cohort, models in groups.items():
                hits = 0
                total = 0
                for model in models:
                    for prompt in range(_N_PROMPTS):
                        margin = self._margin(model, anchor, prompt, seed)
                        total += 1
                        if margin >= threshold:
                            hits += 1
                        elif cohort is _Cohort.STRONG:
                            strong_misses.append(Miss(model, prompt, seed, margin))
                        elif cohort is _Cohort.SFT:
                            sft_misses.append(Miss(model, prompt, seed, margin))
                rates[cohort] = (hits, total)
            false_positives = sum(
                self._margin(test, anchor, prompt, seed) >= threshold
                for prompt in range(_N_PROMPTS)
            )
            heldout.append(SeedRates(
                seed=seed,
                strong_tpr=rates[_Cohort.STRONG][0] / rates[_Cohort.STRONG][1],
                sft_tpr=rates[_Cohort.SFT][0] / rates[_Cohort.SFT][1],
                deploy_fpr=false_positives / _N_PROMPTS,
                hard_tpr=rates[_Cohort.HARD][0] / rates[_Cohort.HARD][1],
            ))
        return DirectionResult(
            anchor=anchor,
            test=test,
            threshold=threshold,
            heldout=(heldout[0], heldout[1]),
            strong_misses=tuple(strong_misses),
            sft_misses=tuple(sft_misses),
        )

    def decision(self) -> LineageDecision:
        """Return the kept pass/fail. Does not accept a table, N, or anchor.

        On the even4 corpus only: for each direction, margin is
        ``sim(model) - sim(anchor)`` on the same prompt and seed. The score
        is not ``sim - max(both cross-family)``. ``passed`` requires, on
        seeds 1 and 2, in both directions, strong TPR >= 0.90, SFT TPR >= 0.90,
        and deploy FPR <= 0.10. Comparison against the threshold is ``>=``.
        Hard TPR is reported and does not gate. A non-even4 corpus returns
        ``passed`` false and ``directions`` None.
        """
        if not self._allow_pass:
            return LineageDecision(passed=False, directions=None)
        qwen, llama = _CROSS
        anchor_qwen = self._direction(qwen, llama)
        anchor_llama = self._direction(llama, qwen)
        directions = BothDirections(
            setting=DecisionSetting(
                layers=_SETTING[0],
                n=_SETTING[1],
                dims=_SETTING[2],
                calibrate_seed=_CALIBRATE_SEED,
                heldout_seeds=_HELDOUT_SEEDS,
            ),
            anchor_qwen=anchor_qwen,
            anchor_llama=anchor_llama,
        )
        passed = all(
            seed.strong_tpr >= _TPR_MIN
            and seed.sft_tpr >= _TPR_MIN
            and seed.deploy_fpr <= _FPR_MAX
            for direction in (anchor_qwen, anchor_llama)
            for seed in direction.heldout
        )
        return LineageDecision(passed=passed, directions=directions)

    def win_counts(self) -> tuple[PromptWinCount, ...]:
        """Return descriptive seed-0 wins. Does not accept names or negatives.

        Order is victim, then SFT, then hard, then strong. Within a model,
        N=20 then 128, dims=8 then 20. A prompt is a win only when the
        similarity is strictly greater than both cross-family similarities.
        These counts do not affect ``decision().passed``.
        """
        counts: list[PromptWinCount] = []
        qwen, llama = _CROSS
        for model in _WIN_MODELS:
            for n in _WIN_NS:
                for dims in _WIN_DIMS:
                    won = 0
                    for prompt in range(_N_PROMPTS):
                        score = self._sim(model, prompt, _CALIBRATE_SEED, n, dims)
                        rival = max(
                            self._sim(qwen, prompt, _CALIBRATE_SEED, n, dims),
                            self._sim(llama, prompt, _CALIBRATE_SEED, n, dims),
                        )
                        if score > rival:
                            won += 1
                    counts.append(PromptWinCount(
                        model=model,
                        role=_PUBLIC_ROLE[_ROLE_OF[model]],
                        n=n,
                        dims=dims,
                        won=won,
                        of=_N_PROMPTS,
                    ))
        return tuple(counts)
