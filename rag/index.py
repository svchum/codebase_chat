
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import ingest
from .config import Config
from .parse import Chunk, chunk_file

try:
    import faiss  
except ImportError:  
    faiss = None

Progress = Callable[[str, int, Optional[int]], None]


# Vector store


class VectorStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._faiss = None
        self._ids = np.zeros((0,), dtype=np.int64)   
        self._mat: Optional[np.ndarray] = None
        self.dim: Optional[int] = None
        self._load()

    # persistence
    @property
    def _file(self) -> Path:
        return self.path.with_suffix(".faiss" if faiss is not None else ".npz")

    def _load(self):
        f = self._file
        if not f.exists():
            return
        try:
            if faiss is not None:
                self._faiss = faiss.read_index(str(f))
                self.dim = self._faiss.d
            else:
                d = np.load(f)
                self._ids, self._mat = d["ids"], d["mat"]
                self.dim = self._mat.shape[1] if self._mat.size else None
        except Exception:
            self.clear()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if faiss is not None and self._faiss is not None:
            faiss.write_index(self._faiss, str(self._file))
        elif faiss is None and self._mat is not None:
            np.savez(self._file, ids=self._ids, mat=self._mat)

    def clear(self):
        self._faiss, self._mat, self.dim = None, None, None
        self._ids = np.zeros((0,), dtype=np.int64)
        for suffix in (".faiss", ".npz"):
            self.path.with_suffix(suffix).unlink(missing_ok=True)

    # ops
    @property
    def count(self) -> int:
        if faiss is not None:
            return int(self._faiss.ntotal) if self._faiss is not None else 0
        return int(len(self._ids))

    def add(self, ids: np.ndarray, vecs: np.ndarray):
        if len(ids) == 0:
            return
        ids = np.asarray(ids, dtype=np.int64)
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        if self.dim is None:
            self.dim = vecs.shape[1]
        if faiss is not None:
            if self._faiss is None:
                self._faiss = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
            self._faiss.add_with_ids(vecs, ids)
        else:
            self._mat = vecs if self._mat is None else np.vstack([self._mat, vecs])
            self._ids = np.concatenate([self._ids, ids])

    def remove(self, ids: list[int]):
        if not ids:
            return
        arr = np.asarray(ids, dtype=np.int64)
        if faiss is not None:
            if self._faiss is not None:
                self._faiss.remove_ids(arr)
        elif self._mat is not None:
            keep = ~np.isin(self._ids, arr)
            self._ids, self._mat = self._ids[keep], self._mat[keep]

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.count == 0 or k <= 0:
            return []
        k = min(k, self.count)
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32)
        if faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        sims = self._mat @ q[0]
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(self._ids[i]), float(sims[i])) for i in top]



# Records and filters


@dataclass
class ChunkRow:
    id: int
    file_path: str
    language: str
    kind: str
    name: str
    parent: str
    start_line: int
    end_line: int
    content: str
    mtime: float
    commit_hash: str

    @property
    def location(self) -> str:
        if self.kind == "commit":
            return self.file_path
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def embed_text(self) -> str:
        return Chunk(self.file_path, self.language, self.kind, self.name, self.parent,
                     self.start_line, self.end_line, self.content).embed_text


@dataclass
class Filters:
    languages: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    name_contains: str = ""
    path_contains: str = ""
    modified_after: Optional[float] = None 

    @property
    def active(self) -> bool:
        return bool(self.languages or self.kinds or self.name_contains.strip()
                    or self.path_contains.strip() or self.modified_after)

    def to_sql(self, alias: str = "") -> tuple[str, list]:
        p = f"{alias}." if alias else ""
        clauses: list[str] = []
        params: list = []
        if self.languages:
            clauses.append(f"{p}language IN ({','.join('?' * len(self.languages))})")
            params += self.languages
        if self.kinds:
            clauses.append(f"{p}kind IN ({','.join('?' * len(self.kinds))})")
            params += self.kinds
        if self.name_contains.strip():
            clauses.append(f"{p}name LIKE ?")
            params.append(f"%{self.name_contains.strip()}%")
        if self.path_contains.strip():
            clauses.append(f"{p}file_path LIKE ?")
            params.append(f"%{self.path_contains.strip()}%")
        if self.modified_after:
            clauses.append(f"{p}mtime >= ?")
            params.append(float(self.modified_after))
        return (" AND ".join(clauses), params)


@dataclass
class SyncStats:
    files_added: int = 0
    files_updated: int = 0
    files_removed: int = 0
    files_unchanged: int = 0
    commits_added: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    rebuilt: bool = False
    seconds: float = 0.0

    def summary(self) -> str:
        return (f"{self.files_added} new / {self.files_updated} changed / {self.files_removed} removed / "
                f"{self.files_unchanged} unchanged files, {self.commits_added} new commits, "
                f"+{self.chunks_added} / -{self.chunks_removed} chunks in {self.seconds:.1f}s"
                + (" (full rebuild)" if self.rebuilt else ""))


_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or", "how", "what",
         "where", "which", "does", "do", "me", "show", "this", "that", "it", "with", "from", "by", "be", "as",
         "at", "about", "can", "i", "my", "we", "you", "there", "any", "all"}


def split_identifier(s: str) -> str:
    """'getUserAuth_token' -> 'get User Auth token' (helps keyword search on code names)."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return re.sub(r"[_\W]+", " ", s).strip()


def fts_query(text: str) -> str:
    toks: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9_]+", text):
        for t in split_identifier(raw).split():
            t = t.lower()
            if len(t) > 1 and t not in _STOP and t not in toks:
                toks.append(t)
    return " OR ".join(f'"{t}"' for t in toks[:24])



# Index


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_path TEXT NOT NULL, language TEXT, kind TEXT, name TEXT, parent TEXT,
  start_line INTEGER, end_line INTEGER, content TEXT, mtime REAL, commit_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path, start_line);
CREATE INDEX IF NOT EXISTS idx_chunks_name ON chunks(name);
CREATE INDEX IF NOT EXISTS idx_chunks_commit ON chunks(commit_hash);
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, sha1 TEXT, mtime REAL);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, name, file_path, tokenize='porter unicode61');
"""

_COLS = "id,file_path,language,kind,name,parent,start_line,end_line,content,mtime,commit_hash"


class Index:
    def __init__(self, index_dir: Path):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.dir / "meta.db", check_same_thread=False)
        self.db.executescript(_SCHEMA)
        self.vs = VectorStore(self.dir / "vectors")

    #  meta 
    def get_meta(self, key: str, default: str = "") -> str:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def set_meta(self, key: str, value: str):
        self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, value))

    def close(self):
        """Release the database file (needed before the folder can be deleted on Windows)."""
        self.db.close()

    def chunk_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def is_consistent(self) -> bool:
        return self.vs.count == self.chunk_count()

    def is_empty(self) -> bool:
        return self.chunk_count() == 0

    def reset(self):
        self.db.executescript("DELETE FROM chunks; DELETE FROM files; DELETE FROM meta; DELETE FROM chunks_fts;")
        self.db.commit()
        self.vs.clear()

    def stats(self) -> dict:
        return {
            "chunks": self.chunk_count(),
            "files": self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0],
            "commits": self.db.execute("SELECT COUNT(*) FROM chunks WHERE kind='commit'").fetchone()[0],
            "last_commit": self.get_meta("last_indexed_commit"),
            "repo_root": self.get_meta("repo_root"),
            "github_base": self.get_meta("github_base"),
            "indexed_at": self.get_meta("indexed_at"),
        }

    def facets(self) -> dict[str, list[str]]:
        langs = [r[0] for r in self.db.execute("SELECT DISTINCT language FROM chunks ORDER BY 1") if r[0]]
        kinds = [r[0] for r in self.db.execute("SELECT DISTINCT kind FROM chunks ORDER BY 1") if r[0]]
        return {"languages": langs, "kinds": kinds}

    # reads
    @staticmethod
    def _row(r) -> ChunkRow:
        return ChunkRow(*r)

    def get_chunks(self, ids: list[int]) -> dict[int, ChunkRow]:
        out: dict[int, ChunkRow] = {}
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            q = f"SELECT {_COLS} FROM chunks WHERE id IN ({','.join('?' * len(part))})"
            for r in self.db.execute(q, part):
                out[r[0]] = self._row(r)
        return out

    def keyword_search(self, query: str, k: int, filters: Optional[Filters] = None) -> list[tuple[int, float]]:
        q = fts_query(query)
        if not q:
            return []
        where, params = (filters.to_sql("c") if filters else ("", []))
        sql = ("SELECT chunks_fts.rowid, bm25(chunks_fts, 1.0, 5.0, 2.0) AS s FROM chunks_fts "
               "JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ?"
               + (f" AND {where}" if where else "") + " ORDER BY s LIMIT ?")
        try:
            rows = self.db.execute(sql, [q, *params, k]).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(r[0]), -float(r[1])) for r in rows]  

    def by_names(self, names: list[str], k: int, filters: Optional[Filters] = None) -> list[int]:
        names = [n for n in names if n]
        if not names:
            return []
        where, params = (filters.to_sql("") if filters else ("", []))
        sql = (f"SELECT id FROM chunks WHERE LOWER(name) IN ({','.join('?' * len(names))})"
               + (f" AND {where}" if where else "") + " LIMIT ?")
        return [r[0] for r in self.db.execute(sql, [*[n.lower() for n in names], *params, k])]

    def allowed_ids(self, filters: Filters, limit: int = 200_000) -> set[int]:
        where, params = filters.to_sql("")
        sql = "SELECT id FROM chunks" + (f" WHERE {where}" if where else "") + " LIMIT ?"
        return {r[0] for r in self.db.execute(sql, [*params, limit])}

    def neighbors(self, row: ChunkRow) -> list[ChunkRow]:
        """Previous/next chunk in the same file, plus the enclosing class header chunk if any."""
        out: list[ChunkRow] = []
        if row.kind == "commit":
            return out
        prev = self.db.execute(
            f"SELECT {_COLS} FROM chunks WHERE file_path=? AND start_line<? ORDER BY start_line DESC LIMIT 1",
            (row.file_path, row.start_line)).fetchone()
        nxt = self.db.execute(
            f"SELECT {_COLS} FROM chunks WHERE file_path=? AND start_line>? ORDER BY start_line ASC LIMIT 1",
            (row.file_path, row.start_line)).fetchone()
        if row.parent:
            par = self.db.execute(
                f"SELECT {_COLS} FROM chunks WHERE file_path=? AND kind='class' AND name=? AND id<>? "
                "ORDER BY start_line LIMIT 1", (row.file_path, row.parent, row.id)).fetchone()
            if par:
                out.append(self._row(par))
        for r in (prev, nxt):
            if r:
                out.append(self._row(r))
        return out

    # writes 
    def _remove_file_chunks(self, path: str) -> int:
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE file_path=?", (path,))]
        if ids:
            self.vs.remove(ids)
            self.db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
            self.db.execute("DELETE FROM chunks WHERE file_path=?", (path,))
        return len(ids)

    def _insert_chunks(self, chunks: list[Chunk], mtimes: list[float], commit_hashes: list[str],
                       embedder, progress: Optional[Progress]) -> int:
        if not chunks:
            return 0
        vecs = embedder.embed_documents(
            [c.embed_text for c in chunks],
            progress=(lambda d, t: progress("embedding", d, t)) if progress else None)
        ids: list[int] = []
        for c, mt, ch in zip(chunks, mtimes, commit_hashes):
            cur = self.db.execute(
                "INSERT INTO chunks(file_path,language,kind,name,parent,start_line,end_line,content,mtime,commit_hash) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (c.file_path, c.language, c.kind, c.name, c.parent, c.start_line, c.end_line, c.content, mt, ch))
            rid = cur.lastrowid
            ids.append(rid)
            self.db.execute("INSERT INTO chunks_fts(rowid,content,name,file_path) VALUES(?,?,?,?)",
                            (rid, c.content, f"{c.name} {split_identifier(c.name)}".strip(),
                             f"{c.file_path} {split_identifier(c.file_path)}"))
        self.vs.add(np.asarray(ids, dtype=np.int64), vecs)
        return len(chunks)

    def sync(self, root: Path, embedder, cfg: Config, progress: Optional[Progress] = None,
             full: bool = False) -> SyncStats:
        """Bring the index in line with the repo on disk. Incremental unless `full` (or the index is stale)."""
        t0 = time.time()
        stats = SyncStats()
        root = Path(root).resolve()

        needs_reset = (
            full
            or not self.is_consistent()
            or (self.get_meta("embed_model") not in ("", cfg.embed_model))
            or (self.get_meta("repo_root") not in ("", str(root)))
        )
        if needs_reset and not self.is_empty():
            self.reset()
            stats.rebuilt = True
        elif needs_reset:
            self.reset()

        known = {r[0]: r[1] for r in self.db.execute("SELECT path, sha1 FROM files")}
        head = ingest.head_commit(root) or ""
        seen: set[str] = set()

        pending: list[Chunk] = []
        pend_mt: list[float] = []
        pend_ch: list[str] = []
        pend_files: list[tuple[str, str, float]] = []

        def flush():
            nonlocal pending, pend_mt, pend_ch, pend_files
            if not pending and not pend_files:
                return
            stats.chunks_added += self._insert_chunks(pending, pend_mt, pend_ch, embedder, progress)
            self.db.executemany(
                "INSERT INTO files(path,sha1,mtime) VALUES(?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET sha1=excluded.sha1, mtime=excluded.mtime", pend_files)
            self.db.commit()
            pending, pend_mt, pend_ch, pend_files = [], [], [], []

        n_seen = 0
        for sf in ingest.iter_source_files(root, cfg.max_file_bytes):
            seen.add(sf.path)
            n_seen += 1
            if progress and n_seen % 50 == 0:
                progress("scanning", n_seen, None)
            if known.get(sf.path) == sf.sha1:
                stats.files_unchanged += 1
                continue
            if sf.path in known:
                stats.chunks_removed += self._remove_file_chunks(sf.path)
                stats.files_updated += 1
            else:
                stats.files_added += 1
            chunks = chunk_file(sf.path, sf.language, sf.text, sf.is_doc, cfg.max_chunk_chars, cfg.overlap_lines)
            pending += chunks
            pend_mt += [sf.mtime] * len(chunks)
            pend_ch += [head] * len(chunks)
            pend_files.append((sf.path, sf.sha1, sf.mtime))
            if len(pending) >= 256:
                flush()
        flush()

        for path in set(known) - seen:
            stats.chunks_removed += self._remove_file_chunks(path)
            self.db.execute("DELETE FROM files WHERE path=?", (path,))
            stats.files_removed += 1
        self.db.commit()


        have = {r[0] for r in self.db.execute("SELECT commit_hash FROM chunks WHERE kind='commit'")}
        new_c: list[Chunk] = []
        new_mt: list[float] = []
        new_ch: list[str] = []
        for cr in ingest.iter_commits(root, cfg.max_commits):
            if cr.sha in have:
                continue
            first = cr.message.strip().splitlines()[0] if cr.message.strip() else ""
            new_c.append(Chunk(f"commit:{cr.sha[:8]}", "git", "commit", first[:120], "", 0, 0, cr.text))
            new_mt.append(cr.date)
            new_ch.append(cr.sha)
        if new_c:
            if progress:
                progress("commits", len(new_c), len(new_c))
            stats.chunks_added += self._insert_chunks(new_c, new_mt, new_ch, embedder, progress)
            stats.commits_added = len(new_c)

        self.set_meta("embed_model", cfg.embed_model)
        self.set_meta("repo_root", str(root))
        self.set_meta("last_indexed_commit", head)
        self.set_meta("github_base", ingest.github_base_url(root) or "")
        self.set_meta("indexed_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.db.commit()
        self.vs.save()
        stats.seconds = time.time() - t0
        return stats