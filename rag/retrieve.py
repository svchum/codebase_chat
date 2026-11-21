"""Retrieval system: Retriever 1: vector search, Retriever 2: metadata / keyword, Retriever 3: contextual expansion:"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .index import ChunkRow, Filters, Index

RRF_K = 60


@dataclass
class Hit:
    chunk: ChunkRow
    score: float = 0.0                      
    sources: list[str] = field(default_factory=list)  
    expanded: bool = False                  


def extract_identifiers(question: str) -> list[str]:
    """Pull likely symbol names out of a question: `backticked`, snake_case, camelCase, PascalCase, foo()."""
    names: list[str] = []
    for m in re.findall(r"`([^`\s]+)`", question):
        names.append(m.strip("()"))
    for m in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)", question):
        names.append(m)
    for m in re.findall(r"\b([a-z]+(?:_[a-z0-9]+)+|[a-z]+(?:[A-Z][a-z0-9]+)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b", question):
        names.append(m)
    out: list[str] = []
    for n in names:
        n = n.split(".")[-1]
        if n and n not in out:
            out.append(n)
    return out


class Retriever:
    def __init__(self, index: Index, embedder):
        self.index = index
        self.embedder = embedder

    # individual retrievers
    def vector_search(self, query: str, k: int, filters: Optional[Filters] = None) -> list[tuple[int, float]]:
        if self.index.vs.count == 0:
            return []
        qv = self.embedder.embed_query(query)
        if filters and filters.active:
            allowed = self.index.allowed_ids(filters)
            if not allowed:
                return []
            fetch = self.index.vs.count if len(allowed) < 20_000 else min(self.index.vs.count, max(k * 50, 2000))
            res = [(i, s) for i, s in self.index.vs.search(qv, fetch) if i in allowed]
            return res[:k]
        return self.index.vs.search(qv, k)

    def keyword_search(self, query: str, k: int, filters: Optional[Filters] = None) -> list[tuple[int, float]]:
        return self.index.keyword_search(query, k, filters)

    def name_search(self, query: str, k: int, filters: Optional[Filters] = None) -> list[int]:
        return self.index.by_names(extract_identifiers(query), k, filters)

    # fusion 
    def retrieve(self, query: str, k: int = 30, filters: Optional[Filters] = None) -> list[Hit]:
        """Run all first-stage retrievers and merge with Reciprocal Rank Fusion."""
        rankings: dict[str, list[int]] = {
            "vector": [i for i, _ in self.vector_search(query, k, filters)],
            "keyword": [i for i, _ in self.keyword_search(query, k, filters)],
            "name": self.name_search(query, 10, filters),
        }
        weights = {"vector": 1.0, "keyword": 0.8, "name": 1.5}
        fused: dict[int, float] = {}
        found_by: dict[int, list[str]] = {}
        for name, ids in rankings.items():
            for rank, cid in enumerate(ids):
                fused[cid] = fused.get(cid, 0.0) + weights[name] / (RRF_K + rank + 1)
                found_by.setdefault(cid, []).append(name)
        order = sorted(fused, key=fused.get, reverse=True)[:k]
        rows = self.index.get_chunks(order)
        return [Hit(rows[i], fused[i], found_by[i]) for i in order if i in rows]

    # contextual expansion 
    def expand(self, hits: list[Hit], max_expand_hits: int = 3, max_extra: int = 6) -> list[Hit]:
        """Append the enclosing class + neighbouring chunks for the top hits (marked expanded=True)."""
        have = {h.chunk.id for h in hits}
        extra: list[Hit] = []
        for h in hits[:max_expand_hits]:
            for n in self.index.neighbors(h.chunk):
                if n.id in have or len(extra) >= max_extra:
                    continue
                have.add(n.id)
                extra.append(Hit(n, 0.0, ["expansion"], expanded=True))
        return hits + extra
