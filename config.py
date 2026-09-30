"""I-I graph experimentation framework configuration.

Research decisions that are not fixed yet stay here so they can be changed
without touching graph-construction code.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    raw_data_dir: Path = Path("/data/amazon2018")
    categories: tuple[str, ...] = ("Beauty", "Sports_and_Outdoors", "Toys_and_Games")
    cache_dir: Path = Path("./cache")
    output_dir: Path = Path("./outputs")
    # If None, the loader searches several common Amazon-2018 filename patterns.
    interaction_file_template: Optional[str] = None
    metadata_file_template: Optional[str] = None


@dataclass
class EmbeddingConfig:
    # "clip" | "openai" | "precomputed"
    backend: str = "clip"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    openai_text_model: str = "text-embedding-3-small"
    openai_image_model: str = "gpt-4o-mini"
    batch_size: int = 64
    # Used only when backend="precomputed". Expected npz keys:
    # item_ids, text_embeddings, image_embeddings, image_valid_mask(optional)
    precomputed_path: Optional[Path] = None


@dataclass
class CollaborativeGraphConfig:
    # "count" | "degree_aware" | "pmi"
    weighting: str = "count"
    min_cooccurrence: int = 2
    top_k: Optional[int] = 50
    # union: keep edge if either endpoint selects it; mutual: both must select it
    symmetrize: str = "union"


@dataclass
class SemanticGraphConfig:
    top_k: int = 20
    sim_threshold: float = 0.0
    # union | mutual
    symmetrize: str = "union"
    # Avoid allocating a full N x N similarity matrix.
    chunk_size: int = 512


@dataclass
class DenoiseConfig:
    enabled: bool = False
    # This remains an experimental placeholder inspired by cross-modal consistency.
    # "threshold" removes weak edges per modality and then synchronizes removals.
    strategy: str = "threshold"
    consistency_threshold: float = 0.3


@dataclass
class FusionConfig:
    # Recommended path for the current experiment:
    # multimodal_behavior: freeze Behavior I-I topology, score only its edges
    # with Text/Image similarity, and soft-reweight behavior strength.
    #
    # Legacy comparison strategies are kept for ablations:
    # none/separate | semantic_backbone | behavior_backbone | union
    strategy: str = "multimodal_behavior"

    # W_ij = (1-lambda) * B_ij + lambda * M_ij
    # Keep behavior dominant at first, then sweep this value in experiments.
    lambda_multimodal: float = 0.3

    # M_ij definition for the multimodal_behavior strategy:
    # strength_x_consistency | mean | min
    reliability_mode: str = "strength_x_consistency"

    # Raw cosine -> semantic evidence:
    # clip01 clips negatives to 0; shift01 maps [-1,1] to [0,1].
    cosine_transform: str = "clip01"

    # Behavior-score normalization before combining with M_ij:
    # log_max | max | minmax | none
    behavior_normalization: str = "log_max"

    # When either Text or Image is missing:
    # behavior_only | single | zero
    missing_modality: str = "behavior_only"

    # Legacy fusion parameters below are used only by the old strategies.
    behavior_weight: float = 0.5
    semantic_weight: float = 0.5
    # mean | min | max for text/image evidence when both are present
    modality_reduce: str = "mean"


@dataclass
class AnalysisConfig:
    enabled: bool = True
    case_study_items: tuple[str, ...] = ()
    top_n_neighbors: int = 10
    export_edge_evidence: bool = True


@dataclass
class HierarchyConfig:
    # Professor's feedback: first verify whether graph structure is meaningful.
    # Therefore hierarchy is experimental and OFF by default.
    enabled: bool = False
    algorithm: str = "louvain"
    resolution: float = 1.0


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    collaborative: CollaborativeGraphConfig = field(default_factory=CollaborativeGraphConfig)
    semantic: SemanticGraphConfig = field(default_factory=SemanticGraphConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
