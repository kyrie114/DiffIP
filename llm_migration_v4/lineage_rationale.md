# Lineage decision

## Problem

The similarities already live in `llm_migration_v4/scores/*.json`. Measurement was frozen before sparse prunes were relabeled as positives and before the threshold experiments, and the historical `neg_` prefix was kept so old files would still load. The pass the later experiments kept is D's aligned margin on even4, N=128, d=8 only. One cross-family model is the anchor, the other is the test, and both directions have to pass. `derivative_cn` is reported and does not gate. mid2, N=20, an absolute cut, a shifted prompt, and the anchor's mean over prompts are not passes. Distillation is a different question. Printed win counts are not a pass. Callers already have the score files. They must not rescore, must not call ALS, and must not relearn the prefix.

## Usage (caller's view)

```python
from pathlib import Path
from llm_migration_v4.lineage import LineagePanel

def write_d_verdict(scores: Path) -> bool:
    decision = LineagePanel.from_score_dir(scores).decision()
    both = decision.directions
    if both is None:
        return False
    for direction in (both.anchor_qwen, both.anchor_llama):
        first, second = direction.heldout
        print(direction.anchor, direction.test, direction.threshold,
              first.sft_tpr, second.sft_tpr, first.hard_tpr)
    return decision.passed

def write_win_lines(scores: Path) -> None:
    for win in LineagePanel.from_score_dir(scores).win_counts():
        print(f"{win.model} {win.role.value} N={win.n} d={win.dims} {win.won}/{win.of}")

def mid2_is_not_a_pass(scores_mid2: Path) -> None:
    decision = LineagePanel.from_score_dir(scores_mid2).decision()
    if decision.passed or decision.directions is not None:
        raise SystemExit("mid2 must not be presented as a lineage pass")
```

## Shape

`LineagePanel` owns the loaded rows and the decision. `from_score_dir` is the only constructor. `decision()` and `win_counts()` take no table, no anchor, and no N. Nothing public returns a row or a similarity.

The membership list is one private tuple. Exact stems, not a `neg_` prefix test. `neg_pruned_*` are strong positives. Deploy is strong union SFT. The two directions are the two orders of the cross-family pair. An import-time check requires 1 victim, 3 strong, 5 SFT, 1 hard, and 2 cross-family. Load raises unless the `main_*.json` stems match that list.

`decision()` computes margins only when the directory basename is `scores`. The margin is `sim(model) - sim(anchor)` on the same prompt and seed. `passed` is true only when both directions have strong TPR >= 0.90, SFT TPR >= 0.90, and deploy FPR <= 0.10 on seeds 1 and 2. Hard TPR is on the rates and does not gate. Held-out strong and SFT cells below the threshold are returned as misses, so a report can name a shortfall without indexing similarities. Basename `scores_mid2` returns `passed` false and `directions` none. Any other basename raises.

`win_counts()` is seed 0, N in {20, 128}, dims in {8, 20}. A prompt counts only when similarity is strictly above both cross-family models. The records have no pass flag.

The public surface hides the JSON shape, the roster, the prefix, the seed split, the 0.90/0.10 numbers, and the even4 restriction. What stays visible is the path, the two anchors, the thresholds, the held-out rates, the misses, and the win counts. An N or anchor argument would hand the policy back to the caller.

## Synthesis decision

Base is candidate B, `LineagePanel`. Both directions are inside `passed`, so a caller cannot report one anchor as the kept result. The table never leaves the object, so the `neg_` rule and the margin formula are not caller code.

Candidate A, `load_even4_main` plus `decide(table, anchor)`, was the other shape. It lost because one `decide` call looks like the whole result, and the table's cells stay readable. Two grafts came from it. `Miss` lists name the held-out cells under the threshold. The roster is checked at import for the closed counts, so a second hand-written list cannot drift quietly.

The cross-judge did not run. This session cannot put a runner on another model family. Both sketches were produced on the parent model, with the structural constraint assigned in each brief so the shapes would still differ.

Rejected from A: exposing `ScoreTable`, making the caller call `decide` twice, and omitting win counts. Rejected from both: parameters for N, dims, mid2, or a misaligned baseline.

## Tradeoffs accepted

- We accept no mid2 rate table in exchange for a result that cannot be presented as a mid2 pass.
- We accept leaving deploy AUC and the 0.95-quantile threshold in `exp_d_calib_even4.json` in exchange for one pass flag.
- We accept a closed roster that rejects an unexpected `main_*.json` in exchange for one membership list and no prefix rule in caller code.
- We accept that the historical `exp_d_prompt_calib.py` still contains the same arithmetic, as the experiment record, in exchange for not rewriting that script. `test_lineage.py` is what fails if the two disagree. New decision code does not copy the rule again.
- We accept a floor of 0.30 on the fine_tune N=128 d=8 mean, together with the basename `scores`, as the even4 witness. The published means are 0.363 and 0.249. A renamed mid2 tree raises. A rescore of even4 that falls below 0.30 also raises.

## Alternatives considered

- A pure function of a loaded table, with the caller choosing the anchor. The caller holds the grid and must remember to AND both directions. That exposes the table and hides less policy. It lost.
- Public stages that return margins, then a threshold, then rates. The caller reassembles the gate. Rejected.
- `decision(n=, dims=, anchor=)` on the panel. Those knobs are the policy. They would let N=20 or mid2 become a second pass. Rejected.

## Open questions and risks

- The basename-only hole is closed. A mid2 tree renamed `scores` raises because its fine_tune mean is below 0.30. A real even4 directory that is not named `scores` still fails closed.
- Should deploy AUC live on `DirectionResult` as a field that does not gate, or stay only in the experiment JSON?

## Next implementation step

`from_score_dir`, `decision`, and `win_counts` are filled in. `python test_lineage.py` checks both directions against `exp_d_calib_even4.json` and the known seed-0 win counts.
