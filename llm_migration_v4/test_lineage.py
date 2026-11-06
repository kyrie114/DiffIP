"""Check LineagePanel against the stored even4 scores and the D result file."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from lineage import LineageDataError, LineagePanel

ROOT = Path(__file__).resolve().parent


def _direction(panel_direction, stored):
    assert panel_direction.anchor == stored["anchor"]
    assert panel_direction.test == stored["test_negative"]
    assert panel_direction.threshold == stored["operating_points"]["fpr0"]["t"]
    for got, seed_name in zip(panel_direction.heldout, ("1", "2")):
        block = stored["operating_points"]["fpr0"]["by_seed"][seed_name]
        assert got.strong_tpr == block["strong"]["tpr"]
        assert got.sft_tpr == block["sft"]["tpr"]
        assert got.deploy_fpr == block["deploy"]["fpr"]
        assert got.hard_tpr == block["hard"]["tpr"]


def main() -> None:
    panel = LineagePanel.from_score_dir(ROOT / "scores")
    decision = panel.decision()
    assert decision.passed
    assert decision.directions is not None
    stored = json.loads((ROOT / "exp_d_calib_even4.json").read_text())
    by_anchor = {row["anchor"]: row for row in stored["directions"] if row["N"] == 128}
    _direction(decision.directions.anchor_qwen, by_anchor["neg_qwen2_5_7b"])
    _direction(decision.directions.anchor_llama, by_anchor["neg_llama3_8b"])
    again = LineagePanel.from_score_dir(ROOT / "scores").decision()
    assert again == decision

    qwen_sft = decision.directions.anchor_qwen.sft_misses
    assert decision.directions.anchor_qwen.heldout[0].sft_tpr == 1.0
    assert len([m for m in qwen_sft if m.seed == 1]) == 0
    assert len([m for m in qwen_sft if m.seed == 2]) == 5

    wins = {(w.model, w.n, w.dims): w.won for w in panel.win_counts()}
    assert wins[("fine_tune", 128, 8)] == 32
    assert wins[("derivative_cn", 128, 8)] == 25
    assert wins[("neg_pruned_sparsegpt", 128, 8)] == 32
    assert wins[("victim", 128, 8)] == 32

    mid = LineagePanel.from_score_dir(ROOT / "scores_mid2").decision()
    assert mid.passed is False
    assert mid.directions is None

    try:
        LineagePanel.from_score_dir(ROOT / "scores_break")
    except LineageDataError:
        pass
    else:
        raise SystemExit("scores_break should not load")

    parent = Path(tempfile.mkdtemp(prefix="lineage-mid2-"))
    try:
        disguised = parent / "scores"
        disguised.mkdir()
        for item in (ROOT / "scores_mid2").glob("main_*.json"):
            os.symlink(item.resolve(), disguised / item.name)
        try:
            LineagePanel.from_score_dir(disguised)
        except LineageDataError:
            pass
        else:
            raise SystemExit("mid2 rows in a directory named scores must not load as even4")
    finally:
        shutil.rmtree(parent)

    print("lineage checks passed")


if __name__ == "__main__":
    main()
