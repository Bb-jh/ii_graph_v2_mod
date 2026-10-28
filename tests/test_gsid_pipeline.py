import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsid.config import load_config
from gsid.stage5_sid import assign_dedup, build_tokens


class SIDTests(unittest.TestCase):
    def test_dedup_and_vocab_separation(self):
        codes = {'T': np.array([[0, 1, 2], [0, 1, 2], [3, 3, 3]]), 'V': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])}
        dedup = assign_dedup(np.concatenate([codes['T'], codes['V']], 1))
        np.testing.assert_array_equal(dedup, [0, 1, 0])
        tokens, vocab = build_tokens(codes, ('T', 'V'), 4, dedup)
        self.assertEqual(len(np.unique(tokens, axis=0)), 3)
        self.assertNotEqual(tokens[2, 0], tokens[2, 1])  # same code, different level -> different token
        self.assertEqual(vocab['vocab_size'], 5 + 6*4 + 2)

    def test_overrides_and_no_graph_order(self):
        cfg = load_config(None, ['guidance.components=["C","R"]', 'sid.use_graph=false', 'tokenizer.lr=0.01'])
        self.assertEqual(cfg.guidance.components, ('C', 'R'))
        self.assertEqual(cfg.sid.order, ('T', 'V'))
        self.assertEqual(cfg.tokenizer.lr, 0.01)
        with self.assertRaises(KeyError):
            load_config(None, ['tokenizer.nope=1'])


class PipelineSmokeTest(unittest.TestCase):
    def test_end_to_end_and_upstream_ablation(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            base = [sys.executable, str(ROOT / 'run_pipeline.py'), '--config', str(ROOT / 'configs/smoke.json')]
            done = subprocess.run(base + ['--output', str(p / 'main'), '--synthetic', str(p / 'data')],
                                  capture_output=True, text=True, timeout=600, cwd=ROOT)
            self.assertEqual(done.returncode, 0, done.stdout[-3000:] + done.stderr[-3000:])
            metrics = json.loads((p / 'main/s6_rec/metrics.json').read_text())
            self.assertEqual(metrics['test']['all']['count'], 300)
            self.assertGreater(metrics['test']['cold']['count'], 0)
            ablation = subprocess.run(base + ['--output', str(p / 'nog'), '--upstream', str(p / 'main'),
                                              '--synthetic', str(p / 'data'), '--stages', '4-5',
                                              '--set', 'sid.use_graph=false', '--set', 'guidance.mode="fixed"'],
                                      capture_output=True, text=True, timeout=600, cwd=ROOT)
            self.assertEqual(ablation.returncode, 0, ablation.stdout[-3000:] + ablation.stderr[-3000:])
            stats = json.loads((p / 'nog/s5_sid/stats.json').read_text())
            self.assertEqual(stats['tokens_per_item'], 7)


if __name__ == '__main__':
    unittest.main()
