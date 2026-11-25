"""End-to-end orchestration. The engine holds several project indexes """
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from . import ingest
from .config import Config
from .embed import Embedder
from .index import Filters, Index, SyncStats
from .llm import LlmTimeout, get_llm, is_busy_error, stream_answer
from .projects import ProjectStore, make_slug
from .prompt import build_messages
from .retrieve import Hit, Retriever
from .usage import UsageCounter


@dataclass
class Engine:
    cfg: Config
    embedder: Embedder
    projects: ProjectStore
    usage: UsageCounter
    _indexes: dict = field(default_factory=dict)  
    _llms: dict = field(default_factory=dict)     
    _last_fetch: float = 0.0

    @classmethod
    def create(cls, cfg: Config) -> "Engine":
        return cls(cfg, Embedder(cfg.embed_model), ProjectStore(cfg.index_dir, cfg.max_projects),
                   UsageCounter(cfg.index_dir / "usage.db"))

    def index_for(self, slug: str) -> Index:
        """The index of one project (opened the first time it is needed)."""
        if slug not in self._indexes:
            self._indexes[slug] = Index(self.projects.path(slug))
        return self._indexes[slug]

    # indexing
    def index_repo(self, source: str, progress=None, full: bool = False) -> SyncStats:
        """Index (or update) a repo. It becomes the newest project; the oldest is dropped if there are too many."""
        if not source.strip():
            raise ValueError("Enter a GitHub URL or a folder path to index.")
        root = ingest.prepare_repo(source, self.cfg.repos_dir)
        slug, name = make_slug(source)
        stats = self.index_for(slug).sync(root, self.embedder, self.cfg, progress, full)
        for old_slug in self.projects.register(slug, name, source):
            self._forget(old_slug)
        return stats

    def _forget(self, slug: str) -> None:
        """Delete a project's index folder."""
        index = self._indexes.pop(slug, None)
        if index:
            index.close()
        shutil.rmtree(self.projects.path(slug), ignore_errors=True)

    def _is_our_clone(self, root: Path) -> bool:
        """True only for repos the app cloned itself (inside REPOS_DIR). A folder you work in is never pulled."""
        try:
            return self.cfg.repos_dir.resolve() in root.resolve().parents
        except OSError:
            return False

    def has_new_commits(self, slug: str, check_remote: bool = True) -> bool:
        """True if the project's repo has commits we haven't indexed (optionally pulls first)."""
        index = self.index_for(slug)
        root = index.get_meta("repo_root")
        if not root or not Path(root).is_dir():
            return False
        if check_remote and self._is_our_clone(Path(root)) and time.time() - self._last_fetch > 60:
            ingest.fetch_remote_updates(Path(root))
            self._last_fetch = time.time()
        head = ingest.head_commit(Path(root))
        return bool(head) and head != index.get_meta("last_indexed_commit")

    def auto_sync(self, slug: str, progress=None) -> Optional[SyncStats]:
        if self.has_new_commits(slug):
            index = self.index_for(slug)
            return index.sync(Path(index.get_meta("repo_root")), self.embedder, self.cfg, progress)
        return None

    # retrieval 
    def retrieve(self, question: str, slug: str, filters: Optional[Filters] = None,
                 top_k: Optional[int] = None, expand: bool = True) -> list[Hit]:
        top_k = top_k or self.cfg.top_k
        retriever = Retriever(self.index_for(slug), self.embedder)
        top = retriever.retrieve(question, k=max(30, top_k * 3), filters=filters)[:top_k]
        return retriever.expand(top) if expand else top

    # generation 
    def _llm_for(self, model: Optional[str] = None):
        """The AI client for the main model (model=None) or a named backup. Each is created once and reused."""
        if model not in self._llms:
            self._llms[model] = get_llm(self.cfg, model)
        return self._llms[model]

    def answer_stream(self, question: str, hits: list[Hit], history: Optional[list[dict]] = None,
                      on_retry=None, on_fallback=None) -> tuple[Iterator[str], list[Hit]]:
        """Returns (token iterator, hits in the order they are numbered in the prompt).

        `on_retry(attempt, total)` is called if the model is busy and the request is being retried.
        `on_fallback(model_name)` is called if the backup model (LLM_FALLBACK_MODEL) takes over.
        """
        messages, used = build_messages(question, hits, history)
        return self._answer_with_backup(messages, on_retry, on_fallback), used

    def _answer_with_backup(self, messages: list[dict], on_retry, on_fallback) -> Iterator[str]:
        backup = self.cfg.llm_fallback_model
        wait = self.cfg.llm_wait_seconds
        started = False
        try:

            for text in stream_answer(self._llm_for(), messages, retries=1 if backup else 3,
                                      on_retry=on_retry, wait_seconds=wait):
                started = True
                yield text
            return
        except Exception as error: 
            if started or not backup or not (is_busy_error(error) or isinstance(error, LlmTimeout)):
                raise
        if on_fallback:
            on_fallback(backup)
        yield from stream_answer(self._llm_for(backup), messages, retries=1, on_retry=on_retry, wait_seconds=wait)