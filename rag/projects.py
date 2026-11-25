"""Keeps track of the last few indexed projects"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Project:
    slug: str        
    name: str         
    source: str      
    indexed_at: str   


def make_slug(source: str) -> tuple[str, str]:
    """Turn a repo URL or folder path into (slug, display name). The same source always gives the same slug."""
    source = source.strip()
    if re.match(r"^(https?://|git@|ssh://)", source):
        cleaned = re.sub(r"\.git$", "", source.rstrip("/"))
        parts = [p for p in re.split(r"[/:]", cleaned) if p]
        name = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        key = cleaned.lower()
    else:
        path = Path(source).expanduser().resolve()
        name, key = path.name or "repo", str(path)
    readable = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "repo"
    digest = hashlib.sha1(key.encode()).hexdigest()[:6]   
    return f"{readable[:40]}-{digest}", name


class ProjectStore:
    def __init__(self, root: Path, max_projects: int = 3):
        self.root = Path(root)
        self.max_projects = max_projects
        self.root.mkdir(parents=True, exist_ok=True)
        self._file = self.root / "projects.json"

    def path(self, slug: str) -> Path:
        return self.root / slug

    def list(self) -> list[Project]:
        """The kept projects, newest first (only ones whose index still exists on disk)."""
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))["projects"]
            projects = [Project(**item) for item in raw]
        except (OSError, ValueError, KeyError, TypeError):
            return []
        return [p for p in projects if (self.path(p.slug) / "meta.db").exists()]

    def get(self, slug: str) -> Optional[Project]:
        return next((p for p in self.list() if p.slug == slug), None)

    def register(self, slug: str, name: str, source: str) -> list[str]:
        """Record that a project was just indexed, making it the newest.
        Returns the slugs that no longer fit in the list (the oldest ones), so the caller can delete them."""
        kept = [p for p in self.list() if p.slug != slug]
        kept.insert(0, Project(slug, name, source, time.strftime("%Y-%m-%d %H:%M")))
        dropped = [p.slug for p in kept[self.max_projects:]]
        self._save(kept[:self.max_projects])
        return dropped

    def _save(self, projects: list[Project]) -> None:
        temp = self._file.with_suffix(".tmp")
        temp.write_text(json.dumps({"projects": [asdict(p) for p in projects]}, indent=2), encoding="utf-8")
        temp.replace(self._file)  