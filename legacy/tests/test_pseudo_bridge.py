import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from old_hierarchy.pseudo_bridge import export_training_labels, prepare_graph, read_graph


def example_graph():
    # LCC 0--1--2--3--4--5, separate component 6--7, isolate 8.
    u = np.array([0, 1, 2, 3, 4, 6])
    v = np.array([1, 2, 3, 4, 5, 7])
    w = np.array([.2, .3, .6, .9, .4, .7], dtype=np.float32)
    return sparse.csr_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(9, 9))


class PseudoBridgeTests(unittest.TestCase):
    def test_weight_and_node_mapping_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            source = p / "graph.npz"
            sparse.save_npz(source, example_graph())
            g, ids = read_graph(source)
            stats = prepare_graph(g, ids, p / "out")
            self.assertEqual(stats["edges"], 6)
            self.assertEqual(stats["largest_component"], 6)
            self.assertEqual(stats["isolated_nodes"], 1)
            with np.load(p / "out" / "edges.npz") as e:
                self.assertTrue(np.all(e["src"] < e["dst"]))
                a = sparse.csr_matrix((e["weight"], (e["src"], e["dst"])), shape=g.shape)
                np.testing.assert_array_equal((a+a.T).toarray(), g.toarray())

    def test_invalid_graphs_and_no_negative_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            for array in [np.array([[0, 1], [0, 0]]), np.array([[0, -1], [-1, 0]]),
                          np.eye(2), np.zeros((2, 2)), np.array([[0, np.nan], [np.nan, 0]])]:
                sparse.save_npz(p / "g.npz", sparse.csr_matrix(array))
                with self.assertRaises(ValueError):
                    read_graph(p / "g.npz")
            star = sparse.csr_matrix([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
            with self.assertRaisesRegex(ValueError, "negative sampling"):
                prepare_graph(star, np.arange(3), p / "out")

    def test_relative_child_ids_get_unique_targets_and_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            g = example_graph()
            ids = np.array([f"ASIN-{x}" for x in range(9)])
            prepare_graph(g, ids, p)
            np.save(p / "structural_embeddings.npy", np.ones((6, 4)))
            with (p / "hierarchical_labels.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["item_id", "graph_node", "global_label", "local_label", "fine_label"])
                for x, path in enumerate([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)]):
                    writer.writerow([ids[x], x, *path])
            result = export_training_labels(p, g, ids)
            self.assertEqual(result["label_counts"], [2, 4, 6])
            with np.load(p / "pseudo_labels.npz", allow_pickle=False) as data:
                self.assertTrue(np.all(data["labels"][6:] == -1))
                np.testing.assert_array_equal(data["valid_mask"], [1, 1, 1, 1, 1, 1, 0, 0, 0])
                self.assertNotEqual(data["labels"][0, 1], data["labels"][3, 1])
                self.assertEqual(data["path_labels"][0, 1], data["path_labels"][3, 1])
                labeled = data["labels"][:6]
                np.testing.assert_array_equal(data["local_parent"][labeled[:, 1]], labeled[:, 0])
                np.testing.assert_array_equal(data["fine_parent"][labeled[:, 2]], labeled[:, 1])

    def test_actual_upstream_pipeline_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            sparse.save_npz(p / "graph.npz", example_graph())
            source_before = (p / "graph.npz").read_bytes()
            (p / "items.json").write_text(json.dumps({str(x): {"title": f"Toy {x}"} for x in range(9)}))
            command = [sys.executable, str(ROOT / "run_graph_hierarchy.py"), "--graph", str(p / "graph.npz"),
                       "--output", str(p / "out"), "--metadata", str(p / "items.json"), "--smoke", "--cpu",
                       "--dim", "8", "--layers", "1", "--global-k", "2", "--local-k", "2", "--fine-k", "2",
                       "--min-cluster-size", "1", "--threads", "1"]
            done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=120)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            manifest = json.loads((p / "out" / "hierarchy_run.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["labels"]["valid_items"], 6)
            self.assertEqual(manifest["labels"]["unlabeled_items"], 3)
            self.assertTrue((p / "out" / "hierarchy_evaluation.json").exists())
            self.assertTrue((p / "out" / "hierarchy_cases.csv").exists())
            self.assertEqual(source_before, (p / "graph.npz").read_bytes())
            refused = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("will not be overwritten", refused.stderr)


if __name__ == "__main__":
    unittest.main()
