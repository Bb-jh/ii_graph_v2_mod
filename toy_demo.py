"""Small dependency-light smoke test for graph modules (no model download)."""
from config import CollaborativeGraphConfig, FusionConfig, SemanticGraphConfig
from data_loader import Interaction
from evidence import EdgeEvidenceStore
from graph.collaborative import build_collaborative_graph, compute_behavior_evidence
from graph.fusion import fuse_graphs
from graph.semantic import build_semantic_graph, score_candidate_edges
import numpy as np

items = ["A", "B", "C", "D"]
interactions = [
    Interaction("u1", "A", 1, "toy"), Interaction("u1", "B", 1, "toy"),
    Interaction("u2", "A", 1, "toy"), Interaction("u2", "B", 1, "toy"), Interaction("u2", "C", 1, "toy"),
    Interaction("u3", "C", 1, "toy"), Interaction("u3", "D", 1, "toy"),
]
text = np.array([[1,0],[.9,.1],[0,1],[.1,.9]], dtype=np.float32)
image = np.array([[1,0],[.8,.2],[.2,.8],[0,1]], dtype=np.float32)
mask = np.ones(4, dtype=bool)

ev = EdgeEvidenceStore()
ccfg = CollaborativeGraphConfig(min_cooccurrence=1, top_k=2)
scores = compute_behavior_evidence(interactions, ccfg, ev)
collab = build_collaborative_graph(items, scores, ccfg)
scfg = SemanticGraphConfig(top_k=2, chunk_size=2)
tg = build_semantic_graph(items, text, mask, scfg, "text", ev)
ig = build_semantic_graph(items, image, mask, scfg, "image", ev)
score_candidate_edges(list(collab.edges()), items, text, mask, "text", ev)
score_candidate_edges(list(collab.edges()), items, image, mask, "image", ev)
fused = fuse_graphs(items, collab, {"text": tg, "image": ig}, ev, FusionConfig(strategy="behavior_backbone"))
print("collaborative", sorted(collab.edges(data=True)))
print("fused", sorted(fused.edges(data=True)))
print("evidence", ev.records)
