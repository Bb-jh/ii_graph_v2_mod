"""Pipeline configuration. Every undecided spec choice is a field with its recommended default."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import json


@dataclass
class DataCfg:
    data_dir: str = "data/Beauty"
    dataset: str = "Beauty"
    # MACRec layout: {dataset}.{train,valid,test}.inter, {dataset}.item.json, embedding NPYs (row = item ID).
    text_emb: str = "{dataset}.emb-llama-td.npy"
    image_emb: str = "{dataset}.emb-ViT-L-14.npy"
    max_history: int = 20


@dataclass
class GraphCfg:  # Stage 1
    min_cooc: int = 1
    top_k: int = 20
    reliability: str = "mean"  # f(cos_T, cos_V): mean | product | min, cosines clipped to [0,1]
    lambda_mm: float = 0.3


@dataclass
class PseudoCfg:  # Stage 2: LightGCN + weighted-edge BPR, then recursive k-means
    dim: int = 64
    layers: int = 3
    epochs: int = 100
    batch_size: int = 4096
    lr: float = 1e-3
    reg: float = 1e-4
    ks: tuple = (8, 4, 4)  # Global, Local per Global, Fine per Local
    min_cluster_size: int = 5  # k is reduced to n // min_cluster_size for small parents
    kmeans_init: int = 10


@dataclass
class GateCfg:  # Stage 3
    eps: float = 1e-12
    tau_s: tuple = (5.0, 5.0, 5.0)
    compat_temperature: float = 0.1  # spec: same τ as the Stage 4 contrastive
    chunk_size: int = 512


@dataclass
class TokenizerCfg:  # Stage 4
    hidden: tuple = (512, 256, 128)
    latent_dim: int = 32
    codebook_size: int = 256
    levels: int = 3
    beta: float = 0.25
    gnn_hidden: int = 256
    gnn_layers: int = 2
    edge_drop: bool = True
    edge_p: float = 0.2
    node_p: float = 0.1
    epochs: int = 200
    batch_size: int = 1024
    labeled_fraction: float = 0.5  # share of each batch drawn cluster-aware; rest uniform over all items
    per_fine: int = 4
    fines_per_local: int = 2
    locals_per_global: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    tau: float = 0.1
    tau_q: float = 1.0
    ema_momentum: float = 0.9
    lambda_max: float = 0.5
    warmup_start: int = 30  # E_0
    warmup_ramp: int = 50
    log_every: int = 10


@dataclass
class GuidanceCfg:
    mode: str = "adaptive"  # adaptive | fixed | none
    components: tuple = ("C", "M", "R", "N")
    fixed_levels: tuple = (2, 3)  # MACRec-style fixed schedule baseline


@dataclass
class SIDCfg:  # Stage 5
    use_graph: bool = True  # False = ablation without Graph SID (Graph branch is not trained either)
    order: tuple = ("T", "G", "V")


@dataclass
class RecCfg:  # Stage 6: T5 trained from scratch (TIGER-size)
    d_model: int = 128
    d_ff: int = 1024
    d_kv: int = 64
    layers: int = 4
    heads: int = 6
    dropout: float = 0.1
    alignment: bool = True  # explicit T<->V SID translation tasks; Graph SID only in rec input/output
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    warmup_steps: int = 1000
    eval_every: int = 5
    patience: int = 5
    beam: int = 20
    eval_batch_size: int = 64
    max_eval_users: int = 0  # 0 = all; >0 subsamples validation only


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    data: DataCfg = field(default_factory=DataCfg)
    graph: GraphCfg = field(default_factory=GraphCfg)
    pseudo: PseudoCfg = field(default_factory=PseudoCfg)
    gates: GateCfg = field(default_factory=GateCfg)
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    guidance: GuidanceCfg = field(default_factory=GuidanceCfg)
    sid: SIDCfg = field(default_factory=SIDCfg)
    rec: RecCfg = field(default_factory=RecCfg)


def _fill(obj, values, path=""):
    known = {f.name: f for f in fields(obj)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _fill(current, value, f"{path}{key}.")
        elif isinstance(current, tuple):
            setattr(obj, key, tuple(value))
        elif isinstance(current, bool):
            if not isinstance(value, bool):
                raise TypeError(f"{path}{key} expects true/false")
            setattr(obj, key, value)
        else:
            setattr(obj, key, type(current)(value))


def load_config(path=None, overrides=()):
    """JSON file, then dotted overrides like ``guidance.components=["C","R"]`` (JSON-parsed values)."""
    cfg = Config()
    if path:
        with open(path, encoding="utf-8") as stream:
            _fill(cfg, json.load(stream))
    for item in overrides:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        nested = value
        for part in reversed(key.split(".")):
            nested = {part: nested}
        _fill(cfg, nested)
    if cfg.guidance.mode not in ("adaptive", "fixed", "none"):
        raise ValueError("guidance.mode must be adaptive, fixed or none")
    if cfg.graph.reliability not in ("mean", "product", "min"):
        raise ValueError("graph.reliability must be mean, product or min")
    if not cfg.sid.use_graph and "G" in cfg.sid.order:
        cfg.sid.order = tuple(m for m in cfg.sid.order if m != "G")
    return cfg


def to_dict(cfg):
    return asdict(cfg)
