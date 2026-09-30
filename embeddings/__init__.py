from config import EmbeddingConfig


def get_embedder(cfg: EmbeddingConfig):
    if cfg.backend == "clip":
        from embeddings.clip_embedder import ClipEmbedder
        return ClipEmbedder(cfg.clip_model_name, cfg.batch_size)
    if cfg.backend == "openai":
        from embeddings.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(cfg.openai_text_model, cfg.openai_image_model, cfg.batch_size)
    if cfg.backend == "precomputed":
        return None
    raise ValueError(f"Unknown embedding backend: {cfg.backend}")
