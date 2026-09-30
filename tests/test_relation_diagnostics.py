"""Analytical checks and a synthetic end-to-end run (no external data/model)."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_relation_diagnostics import compare_ranks, sample_pairs, scores


class DiagnosticsTests(unittest.TestCase):
    def test_penalty_and_missing_modalities(self):
        cfg = dict(lambda_multimodal=.3, cosine_transform="clip01", missing_modality="behavior_only")
        b = np.array([.5, .5, .5, .5])
        mean, penalty, _, _ = scores(b, np.array([.9, .8, np.nan, np.nan]),
                                     np.array([.3, .8, .7, np.nan]), cfg)
        np.testing.assert_allclose(mean, [.53, .59, .5, .5])
        np.testing.assert_allclose(penalty, [.422, .59, .5, .5])
        cfg["missing_modality"] = "single"
        self.assertAlmostEqual(scores(b, np.full(4, np.nan), np.full(4, .7), cfg)[0][0], .56)
        cfg["missing_modality"] = "zero"
        self.assertAlmostEqual(scores(b, np.full(4, np.nan), np.full(4, .7), cfg)[0][0], .35)

    def test_ties_do_not_count_as_strict_reversal(self):
        edges = np.array([[0, 1], [0, 2], [0, 3]])
        result, nodes, _ = compare_ranks(edges, np.array([2., 2., 1.]), np.array([1., 3., 2.]))
        self.assertEqual(result["strict_neighbor_pairs"], 2)
        self.assertEqual(result["strict_inversions"], 1)
        self.assertEqual(nodes[0]["top_outside_reference_top_tie"], False)

    def test_pairs_unique_reproducible_and_no_self(self):
        for count in [10, 200]:
            a = sample_pairs(np.arange(8), count, 123)
            np.testing.assert_array_equal(a, sample_pairs(np.arange(8), count, 123))
            self.assertEqual(len(a), min(count, 28))
            self.assertEqual(len(set(map(tuple, a))), len(a))
            self.assertTrue(np.all(a[:, 0] < a[:, 1]))

    def test_full_run_reorders_ids_and_excludes_zero_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            source.mkdir()
            cfg = dict(lambda_multimodal=.3, cosine_transform="clip01", missing_modality="behavior_only",
                       reliability_mode="strength_x_consistency", behavior_edges=2,
                       behavior_normalization="log_max", symmetrize="mutual", top_k=20, min_cooc=2)
            (source / "run_config.json").write_text(json.dumps(cfg))
            (source / "graph_analysis.json").write_text(json.dumps({"graphs": {"collaborative": {"nodes": 4}}}))
            b = np.log1p([2., 4.]) / np.log1p(4.)
            t, i = [.8, .6], [0., .8]
            with (source / "edge_evidence.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["item_i", "item_j", "normalized_behavior_score", "text_similarity", "image_similarity", "final_score"])
                for k, (u, v) in enumerate([(0, 1), (1, 2)]):
                    final = .7*b[k] + .3*(t[k]+i[k])/2*(1-abs(t[k]-i[k]))
                    writer.writerow([u, v, b[k], t[k], i[k], final])
            text = np.array([[1., 0], [.8, .6], [0, 1], [-1, 0]])
            image = np.array([[1., 0, 0], [0, 1, 0], [.6, .8, 0], [0, 0, 0]])
            order = [2, 0, 3, 1]
            path = root / "embeddings.npz"
            np.savez(path, item_ids=np.array(order), text_embeddings=text[order], image_embeddings=image[order])
            command = [sys.executable, str(ROOT / "run_relation_diagnostics.py"), "--results", str(source),
                       "--output", str(output), "--embedding-npz", str(path), "--random-pairs", "100"]
            done = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            report = json.loads((output / "diagnostics.json").read_text())
            self.assertEqual(report["step1"]["jointly_valid_items"], 3)
            self.assertEqual(report["step1"]["sampled_pairs"], 3)
            self.assertEqual(report["step1"]["sampled_behavior_edges"], 2)
            self.assertAlmostEqual(report["step1"]["text"]["behavior"]["mean"], .7)
            self.assertTrue((output / "similarity_distributions.pdf").exists())
            # Deliberately wrong feature-to-ID mapping must fail before claiming completion.
            np.savez(path, item_ids=np.array(order), text_embeddings=text, image_embeddings=image[order])
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("do not reproduce archive", failed.stderr)


if __name__ == "__main__":
    unittest.main()
