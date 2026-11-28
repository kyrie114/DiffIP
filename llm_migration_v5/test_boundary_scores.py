"""Shipped QKV unpacking and joint per-layer scores."""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import suite_sota as S  # noqa: E402


class PackedQkvTest(unittest.TestCase):
    def test_interleaved_layout_matches_head_reshape_not_chunk(self):
        n_heads, head_dim, hidden = 32, 128, 4096
        q = torch.arange(n_heads * head_dim * hidden, dtype=torch.float32).reshape(n_heads, head_dim, hidden)
        k = q + 1.0e7
        v = q + 2.0e7
        packed = torch.stack([q, k, v], dim=1).reshape(3 * hidden, hidden)
        model = SimpleNamespace(config=SimpleNamespace(
            hidden_size=hidden,
            num_attention_heads=n_heads,
            head_dim=head_dim,
        ))
        wq, wk, wv = S._split_packed_qkv(packed, model)
        fused = packed.view(n_heads, 3, head_dim, hidden)
        self.assertTrue(torch.equal(wq, fused[:, 0].reshape(hidden, hidden)))
        self.assertTrue(torch.equal(wk, fused[:, 1].reshape(hidden, hidden)))
        self.assertTrue(torch.equal(wv, fused[:, 2].reshape(hidden, hidden)))
        blocked = packed.chunk(3, dim=0)
        self.assertFalse(torch.equal(wq, blocked[0]))
        self.assertFalse(torch.equal(wk, blocked[1]))


class JointLayerScoreTest(unittest.TestCase):
    def test_per_layer_terms_rebuild_score_ours_pair(self):
        rng = np.random.default_rng(0)
        sus = rng.standard_normal((32, 3, 4, 16, 8)).astype(np.float64)
        vic = sus + 0.05 * rng.standard_normal(sus.shape)
        np.random.seed(0)
        raw, _norm = S.per_layer_joint_scores(sus, vic, dims=4)
        np.random.seed(0)
        sims, _elapsed = S.score_ours_pair(sus, vic, dims=4)
        rebuilt = np.exp(-(-np.log(raw)).mean(axis=-1))
        self.assertTrue(np.allclose(rebuilt, sims, rtol=1e-10, atol=1e-10))

        np.random.seed(0)
        alone = np.zeros_like(raw)
        for p in range(32):
            for k in range(3):
                for i in range(4):
                    pr = S.project_sequences(sus[p, k, i:i + 1, :128], vic[p, k, i:i + 1, :128], 4)
                    dist = float(S.fit_step_alignment(pr.suspect[0], pr.victim[0]).distance)
                    alone[p, k, i] = np.exp(-dist)
        self.assertFalse(np.allclose(alone, raw, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
