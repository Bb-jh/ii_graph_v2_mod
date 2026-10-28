import math
import unittest

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from hierarchy.confidence import graph_teacher_confidence
from hierarchy.adaptive_distillation import (
    LossEMA, adaptive_weights, compatibility, contrastive_from_scores, guidance_lambda, hierarchy_guidance,
    hierarchy_masks, level_contrastive_loss, normalize_loss, st_softmax_blocks, weighted_guidance_loss,
)
from hierarchy.sampling import cluster_aware_batches, edge_drop
from scripts.diagnostics.run_adaptive_diagnostics import summarize


class ConfidenceTests(unittest.TestCase):
    def test_weighted_topology_parent_scope_and_missing_labels(self):
        graph = sparse.csr_matrix([[0, 3, 1, 0, 0], [3, 0, 0, 0, 0],
                                   [1, 0, 0, 2, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0]])
        labels = np.array([[0, 0], [0, 1], [1, 2], [1, 3], [-1, -1]])
        result = graph_teacher_confidence(graph, np.array([[0], [1], [9], [10], [0]]), labels)
        self.assertAlmostEqual(result['topology'][0, 0], .75)
        self.assertEqual(result['topology'][0, 1], 0)
        np.testing.assert_array_equal(result['neighbor_count'][0], [2, 1])
        np.testing.assert_array_equal(result['confidence'][4], [0, 0])
        self.assertAlmostEqual(result['assignment'][0, 0], 9/9.5)
        self.assertAlmostEqual(result['support'][0, 0], 1-np.exp(-2/5), places=6)

    def test_single_cluster_is_one_and_negative_margin_is_zero(self):
        graph = sparse.csr_matrix(np.ones((3, 3))-np.eye(3))
        z = np.array([[0.], [10.], [1.]])
        result = graph_teacher_confidence(graph, z, np.array([[0], [0], [1]]))
        self.assertEqual(result['assignment'][0, 0], 0)
        result = graph_teacher_confidence(graph, z, np.zeros((3, 1), dtype=int), tau_s=[2.])
        self.assertTrue(np.all(result['assignment'] == 1))
        np.testing.assert_allclose(result['confidence'], np.cbrt(1-np.exp(-2/2)), rtol=1e-6)


# Global {0,1}; Local {0,1 | 2}; Fine {0,1 | 2 | 3}; row 6 unlabeled.
LABELS = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3], [1, 2, 3], [-1, -1, -1]])


class MaskTests(unittest.TestCase):
    def test_parent_scoped_sets_and_validity(self):
        candidate, positive, valid = hierarchy_masks(LABELS)
        self.assertEqual(candidate[0, 0].nonzero().flatten().tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(positive[0, 0].nonzero().flatten().tolist(), [1, 2, 3])
        self.assertEqual(candidate[0, 1].nonzero().flatten().tolist(), [1, 2, 3])
        self.assertEqual(positive[0, 1].nonzero().flatten().tolist(), [1, 2])
        self.assertEqual(candidate[0, 2].nonzero().flatten().tolist(), [1, 2])
        self.assertEqual(positive[0, 2].nonzero().flatten().tolist(), [1])
        self.assertEqual(valid[0].tolist(), [True, True, True])
        self.assertEqual(valid[3].tolist(), [True, False, False])  # Local 1 has no other member
        self.assertEqual(valid[4].tolist(), [True, False, False])  # Global 1 has a single Local
        self.assertFalse(valid[6].any() or candidate[6].any() or candidate[:, :, 6].any())


class LossTests(unittest.TestCase):
    def test_bounds_of_normalized_loss(self):
        candidate = torch.tensor([[True, True, True, True]])
        positive = torch.tensor([[True, False, False, False]])
        valid = torch.tensor([True])
        random = contrastive_from_scores(torch.zeros(1, 4), candidate, positive, .1)
        self.assertAlmostEqual(random.item(), math.log(4), places=5)
        self.assertAlmostEqual(normalize_loss(random, candidate, positive, valid).item(), 1, places=5)
        perfect = contrastive_from_scores(torch.tensor([[1., -1, -1, -1]]), candidate, positive, .01)
        self.assertAlmostEqual(normalize_loss(perfect, candidate, positive, valid).item(), 0, places=4)
        empty = contrastive_from_scores(torch.zeros(1, 4), torch.zeros_like(candidate), positive, .1)
        self.assertEqual(empty.item(), 0)

    def test_st_softmax_forward_hard_backward_soft(self):
        torch.manual_seed(0)
        z = torch.randn(5, 4, requires_grad=True)
        books = [torch.randn(6, 4, requires_grad=True) for _ in range(3)]
        blocks, codes = st_softmax_blocks(z, books, tau_q=.5)
        residual = z.detach()
        for level, book in enumerate(books):
            expected = (residual[:, None] - book.detach()[None]).square().sum(-1).argmin(-1)
            self.assertTrue(torch.equal(codes[:, level], expected))
            self.assertTrue(torch.allclose(blocks[:, level], F.normalize(book.detach()[expected], dim=-1), atol=1e-6))
            residual = residual - book.detach()[expected]
        candidate, positive, _ = hierarchy_masks(LABELS[:5])
        level_contrastive_loss(blocks, candidate, positive).sum().backward()
        self.assertGreater(z.grad.abs().sum().item(), 0)
        self.assertGreater(books[2].grad.abs().sum().item(), 0)

    def test_compatibility_structured_vs_random(self):
        candidate, positive, valid = hierarchy_masks(LABELS)
        y = LABELS.clamp_min(0)
        structured = torch.eye(9)[y[:, 0]] + torch.eye(9)[y[:, 1]+2] + torch.eye(9)[y[:, 2]+5]
        high = compatibility(structured, candidate, positive, valid, temperature=1.)
        same = compatibility(torch.ones(7, 3), candidate, positive, valid)
        self.assertTrue((high[valid] > .2).all())
        self.assertTrue(torch.allclose(same[valid], torch.zeros_like(same[valid]), atol=1e-5))
        self.assertTrue((high[~valid] == 0).all())


class WeightTests(unittest.TestCase):
    def test_need_readiness_and_gates(self):
        _, _, valid = hierarchy_masks(LABELS)
        ids = torch.arange(7)
        tracker = LossEMA(7, momentum=.5)
        loss = torch.tensor([[1., 2, 3]]).expand(7, -1).clone()
        loss[1] *= 3
        normalized = torch.full((7, 3), .25)
        one = torch.ones(7, 3)
        weights = adaptive_weights(tracker, ids, {'T': loss, 'G': loss}, {'T': normalized, 'G': normalized},
                                   one, {'T': one}, valid)
        need = tracker.need('T', ids, valid)
        for level in range(3):
            self.assertAlmostEqual(need[valid[:, level], level].mean().item(), 1, places=5)
        self.assertGreater(need[1, 0], need[0, 0])
        ready = tracker.readiness('T', ids)
        self.assertTrue(torch.all(ready[:, 0] == 1))
        self.assertAlmostEqual(ready[0, 1].item(), .75)
        self.assertEqual(ready[6, 1].item(), 0)  # parent level never observed
        self.assertTrue(torch.allclose(weights['G'][valid], (one*ready)[valid]))
        self.assertTrue(torch.allclose(weights['T'][valid], (ready*need)[valid]))
        self.assertTrue((weights['T'][~valid] == 0).all())
        tracker.update('T', ids, loss*0, normalized*0, valid)
        self.assertAlmostEqual(tracker.loss[0, 0, 0].item(), .5)  # EMA, not overwrite

    def test_absolute_scale_and_stop_gradient(self):
        torch.manual_seed(1)
        e = {m: F.normalize(torch.randn(7, 3, 4), dim=-1).requires_grad_() for m in 'TVG'}
        compat = {m: torch.rand(7, 3) for m in 'TV'}
        tracker = LossEMA(7)
        conf = torch.ones(7, 3)
        loss, info = hierarchy_guidance(tracker, torch.arange(7), LABELS, e, conf, compat)
        self.assertFalse(any(w.requires_grad for w in info['weights'].values()))
        half = weighted_guidance_loss(info['loss'], {k: v*.5 for k, v in info['weights'].items()}, info['valid'])
        self.assertTrue(torch.allclose(half, loss*.5))
        loss.backward()
        self.assertGreater(e['G'].grad.abs().sum().item(), 0)
        zero, _ = hierarchy_guidance(LossEMA(7), torch.arange(7), LABELS, e, conf*0, compat)
        self.assertEqual(zero.item(), 0)

    def test_component_ablation_and_fixed_schedule(self):
        e = {m: F.normalize(torch.randn(7, 3, 4), dim=-1) for m in 'TVG'}
        zero = {m: torch.zeros(7, 3) for m in 'TV'}
        conf = torch.ones(7, 3)
        _, info = hierarchy_guidance(LossEMA(7), torch.arange(7), LABELS, e, conf, zero, components=('C', 'R', 'N'))
        valid = info['valid']
        self.assertTrue((info['weights']['T'][valid] > 0).any())  # M switched off -> not gated by M=0
        _, info = hierarchy_guidance(LossEMA(7), torch.arange(7), LABELS, e, conf*0, zero, fixed_levels=(2, 3))
        w = info['weights']['V']
        self.assertTrue((w[:, 0] == 0).all())
        self.assertTrue(torch.equal(w[:, 1:], valid[:, 1:].float()))

    def test_lambda_schedule(self):
        self.assertEqual(guidance_lambda(2, start=5, ramp=10), 0)
        self.assertAlmostEqual(guidance_lambda(10, start=5, ramp=10, maximum=.4), .2)
        self.assertEqual(guidance_lambda(99, start=5, ramp=10), 1)


class SamplingTests(unittest.TestCase):
    def test_cluster_aware_batches_are_unique_labeled_and_mostly_valid(self):
        rng = np.random.default_rng(0)
        labels = np.array([[g, g*3+l, (g*3+l)*2+f] for g in range(4) for l in range(3) for f in range(2) for _ in range(5)])
        labels = np.vstack([labels, [[-1, -1, -1]]*3])
        batches = list(cluster_aware_batches(labels, 32, rng=rng))
        self.assertEqual(len(batches), 4)
        for batch in batches:
            self.assertEqual(len(np.unique(batch)), len(batch))
            self.assertTrue((labels[batch] >= 0).all())
            _, _, valid = hierarchy_masks(torch.from_numpy(labels[batch]))
            self.assertGreater(valid.float().mean().item(), .8)

    def test_edge_drop_isolates_nodes(self):
        edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
        kept, weight, isolated = edge_drop(edges, torch.ones(4), 4, edge_p=0, node_p=.5,
                                           generator=torch.Generator().manual_seed(3))
        self.assertEqual(len(weight), kept.shape[1])
        self.assertFalse(torch.isin(kept, isolated.nonzero().flatten()).any())
        kept, _, _ = edge_drop(edges, None, 4, edge_p=0, node_p=0)
        self.assertTrue(torch.equal(kept, edges))


class DiagnosticsTests(unittest.TestCase):
    def test_summary_excludes_invalid_and_zero_baseline(self):
        compat = torch.tensor([[.1], [.9], [0.]])
        valid = torch.tensor([[True], [True], [False]])
        rows = summarize(compat, torch.ones(3, 1), valid)
        self.assertEqual(rows[0]['count'], 2)
        self.assertAlmostEqual(rows[0]['CM_mean'], .5, places=6)
        self.assertAlmostEqual(rows[0]['CM_retained_vs_C'], .5, places=6)
        self.assertIsNone(rows[2]['CM_mean'])
        self.assertIsNone(summarize(compat, torch.zeros(3, 1), valid)[0]['CM_retained_vs_C'])


if __name__ == '__main__':
    unittest.main()
