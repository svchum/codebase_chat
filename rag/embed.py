"""local embeddings with sentence-transformers"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._model.max_seq_length = min(getattr(self._model, "max_seq_length", 512) or 512, 512)
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension  
        return int(get_dim())

    def embed_documents(self, texts: list[str], batch_size: int = 32,
                        progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        model = self._load()
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            out.append(model.encode(batch, normalize_embeddings=True, show_progress_bar=False,
                                    convert_to_numpy=True))
            if progress:
                progress(min(i + batch_size, len(texts)), len(texts))
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(out).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        q = BGE_QUERY_PREFIX + query if "bge" in self.model_name.lower() else query
        v = self._load().encode([q], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return v[0].astype(np.float32)