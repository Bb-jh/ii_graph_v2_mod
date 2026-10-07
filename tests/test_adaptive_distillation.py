import unittest

import numpy as np
from scipy import sparse
import torch

from hierarchy.confidence import graph_teacher_confidence
from hierarchy.adaptive_distillation import (
    adaptive_weights, hierarchy_candidate_mask, weighted_guidance_loss,
)
from run_adaptive_diagnostics import summarize, weight_variants


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
        self.assertAlmostEqual(result['support'][0, 0], 1-np.exp(-2/5))

    def test_single_cluster_and_negative_margin_are_zero(self):
        graph = sparse.csr_matrix(np.ones((3, 3))-np.eye(3))
        z = np.array([[0.], [10.], [1.]])
        result = graph_teacher_confidence(graph, z, np.array([[0], [0], [1]]))
        self.assertEqual(result['assignment'][0, 0], 0)
        result = graph_teacher_confidence(graph, z, np.zeros((3, 1), dtype=int))
        self.assertTrue(np.all(result['confidence'] == 0))


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.g = torch.randn(6, 2, 4, requires_grad=True)
        self.s = torch.randn(6, 2, 3, requires_grad=True)
        self.o = torch.randn(6, 5, requires_grad=True)
        self.c = torch.ones(6, 2)
        self.mask = hierarchy_candidate_mask(torch.tensor([[0, 0], [0, 1], [0, 1], [1, 2], [1, 2], [-1, -1]]))

    def test_identity_empty_and_single_candidates(self):
        result = adaptive_weights(self.g, self.g, self.o, self.c, self.mask)
        self.assertTrue(torch.allclose(result['need'], torch.zeros_like(self.c), atol=1e-6))
        self.assertTrue(torch.isfinite(result['alpha']).all())
        self.assertFalse(result['usable'][5].any())
        self.assertFalse(result['usable'][3, 1])
        self.assertTrue(result['usable'][0, 1])
        same = self.o[:, None, :].expand(-1, 2, -1)
        result = adaptive_weights(same, self.s, self.o, self.c, self.mask)
        self.assertTrue(torch.allclose(result['compatibility'][result['usable']], torch.ones_like(result['compatibility'][result['usable']])))

    def test_stop_gradient_and_absolute_loss_scale(self):
        result = adaptive_weights(self.g, self.s, self.o, self.c, self.mask)
        alpha = result['alpha']
        self.assertFalse(alpha.requires_grad)
        self.assertTrue(((alpha >= 0) & (alpha <= 1)).all())
        loss = self.s.square().mean(-1)
        full = weighted_guidance_loss(loss, alpha, result['usable'])
        half = weighted_guidance_loss(loss, alpha*.5, result['usable'])
        self.assertTrue(torch.allclose(half, full*.5))
        full.backward()
        self.assertGreater(self.s.grad.abs().sum().item(), 0)
        self.assertIsNone(self.g.grad)
        self.assertIsNone(self.o.grad)
        empty = weighted_guidance_loss(loss, alpha, torch.zeros_like(result['usable']))
        self.assertEqual(empty.item(), 0)

    def test_original_does_not_directly_change_need(self):
        first = adaptive_weights(self.g, self.s, self.o, self.c, self.mask)
        second = adaptive_weights(self.g, self.s, self.o.flip(0), self.c, self.mask)
        self.assertTrue(torch.equal(first['need'], second['need']))
        self.assertFalse(torch.allclose(first['compatibility'], second['compatibility']))

    def test_student_original_identity_links_need_and_compatibility(self):
        student = self.o[:, None, :].expand(-1, 2, -1)
        result = adaptive_weights(self.g, student, self.o, self.c, self.mask)
        valid = result['usable']
        self.assertTrue(torch.allclose(result['need'][valid], 1-result['compatibility'][valid], atol=1e-6))
        self.assertTrue((result['alpha'] <= .25+1e-6).all())

    def test_attenuation_report_excludes_unusable_and_handles_zero_baseline(self):
        result = dict(need=torch.tensor([[.9], [.1], [0.]]),
                      compatibility=torch.tensor([[.1], [.9], [0.]]),
                      usable=torch.tensor([[True], [True], [False]]))
        rows = summarize(result, torch.ones(3, 1))
        self.assertEqual(rows[0]['count'], 2)
        self.assertAlmostEqual(rows[0]['CNM_mean'], .09, places=6)
        self.assertAlmostEqual(rows[0]['CNM_retained_vs_CN'], .18, places=6)
        self.assertIsNone(rows[2]['CNM_mean'])
        result['need'].zero_()
        self.assertIsNone(summarize(result, torch.ones(3, 1))[0]['CNM_retained_vs_CN'])
        variants = weight_variants(torch.tensor(1.), torch.tensor(.9), torch.tensor(.1))
        self.assertAlmostEqual(float(variants['CNM']), .09, places=6)
        self.assertGreater(variants['CN_sqrtM'], variants['CNM'])
        self.assertGreater(variants['CN_floorM'], variants['CNM'])


if __name__ == '__main__':
    unittest.main()
