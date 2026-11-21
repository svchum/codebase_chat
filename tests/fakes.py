"""Deterministic fake embedder so tests need no model downloads"""
import re
import zlib

import numpy as np


class FakeEmbedder:
    model_name = "fake"
    dim = 64

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for t in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(t.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts, batch_size=32, progress=None):
        out = np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)
        if progress:
            progress(len(texts), len(texts))
        return out

    def embed_query(self, q):
        return self._vec(q)
