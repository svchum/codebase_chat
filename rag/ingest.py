"""repo acquisition, source & doc file loading, and git commit loading"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pathspec

CODE_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java", ".go": "go",

    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp", ".rs": "rust",
    ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".swift": "swift", ".scala": "scala",
    ".sh": "shell", ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "css", ".json": "json",
    ".vue": "vue", ".svelte": "svelte",
}
DOC_EXT = {".md": "markdown", ".markdown": "markdown", ".rst": "rst", ".txt": "text", ".adoc": "asciidoc", ".pdf": "pdf"}

SKIP_DIRS = {".git", "node_modules", "vendor", "third_party", "dist", "build", "target", "__pycache__",
             ".venv", "venv", "env", ".tox", ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
             ".next", ".gradle", "site-packages", "bower_components"}
SKIP_FILE_PATTERNS = (".min.js", ".min.css", ".bundle.js", "package-lock.json", "yarn.lock", "poetry.lock",
                      "package.json.lock", ".map", "pnpm-lock.yaml", "composer.lock")


def _pdf_text(data: bytes) -> str:
    """Extract text from a PDF (needs `pypdf`; returns "" if unavailable or image-only)."""
    try:
        import io
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    except Exception:
        return ""


@dataclass
class SourceFile:
    path: str         
    language: str
    is_doc: bool
    text: str
    sha1: str
    mtime: float


@dataclass
class CommitRecord:
    sha: str
    author: str
    date: float        
    message: str
    files: list[str]

    @property
    def text(self) -> str:
        files = ", ".join(self.files[:20]) + (" ..." if len(self.files) > 20 else "")
        when = time.strftime("%Y-%m-%d", time.gmtime(self.date))
        return f"Commit {self.sha[:8]} by {self.author} on {when}\n{self.message.strip()}\nFiles changed: {files}"



# Repo acquisition

def _is_url(source: str) -> bool:
    return bool(re.match(r"^(https?://|git@|ssh://)", source.strip()))


def _repo_dirname(url: str) -> str:
    name = re.sub(r"\.git$", "", url.strip().rstrip("/")).split("/")[-1].split(":")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "repo"


def prepare_repo(source: str, repos_dir: Path, history_depth: int = 500) -> Path:
    """Return a local path for `source` (local path used as-is; URLs are cloned/updated)."""
    source = source.strip()
    if not _is_url(source):
        p = Path(source).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"Local repo path not found: {p}")
        return p

    import git  
    repos_dir.mkdir(parents=True, exist_ok=True)
    dest = repos_dir / _repo_dirname(source)
    if (dest / ".git").is_dir():
        repo = git.Repo(dest)
        repo.git.update_environment(GIT_TERMINAL_PROMPT="0")
        try:
            repo.remotes.origin.pull(kill_after_timeout=120)
        except Exception:
            pass  
    else:
        git.Repo.clone_from(source, dest, depth=history_depth)
    return dest.resolve()


def fetch_remote_updates(root: Path) -> None:
    """Best-effort pull for cloned repos (so auto-sync can see new upstream commits)."""
    try:
        import git
        repo = git.Repo(root)
        repo.git.update_environment(GIT_TERMINAL_PROMPT="0")   
        if repo.remotes and not repo.is_dirty():
            repo.remotes.origin.pull(kill_after_timeout=20)   
    except Exception:
        pass



# Git metadata


def head_commit(root: Path) -> Optional[str]:
    try:
        import git
        return git.Repo(root).head.commit.hexsha
    except Exception:
        return None


def github_base_url(root: Path) -> Optional[str]:
    """https://github.com/owner/repo for the repo's origin remote (None if not GitHub)."""
    try:
        import git
        url = git.Repo(root).remotes.origin.url
    except Exception:
        return None
    m = re.match(r"^(?:git@github\.com:|https?://(?:[^@/]+@)?github\.com/|ssh://git@github\.com/)([^/]+)/(.+?)(?:\.git)?/?$", url)
    return f"https://github.com/{m.group(1)}/{m.group(2)}" if m else None


def iter_commits(root: Path, max_commits: int) -> Iterator[CommitRecord]:
    try:
        import git
        repo = git.Repo(root)
        for c in repo.iter_commits(max_count=max_commits):
            try:
                files = list(c.stats.files.keys())
            except Exception:
                files = []
            yield CommitRecord(c.hexsha, str(c.author), float(c.committed_date), str(c.message), files)
    except Exception:
        return



# File walking


def _load_gitignore(root: Path) -> Optional[pathspec.PathSpec]:
    gi = root / ".gitignore"
    if not gi.is_file():
        return None
    try:
        return pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text(errors="ignore").splitlines())
    except Exception:
        return None


def _looks_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return True


def iter_source_files(root: Path, max_file_bytes: int = 1_000_000) -> Iterator[SourceFile]:
    """Yield code and doc files, honoring .gitignore and skipping binaries / vendored dirs."""
    root = Path(root)
    spec = _load_gitignore(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not (spec and spec.match_file(f"{rel_dir}/{d}/".lstrip("/")))
        )
        for fn in sorted(filenames):
            rel = f"{rel_dir}/{fn}".lstrip("/")
            if spec and spec.match_file(rel):
                continue
            if fn.endswith(SKIP_FILE_PATTERNS):
                continue
            ext = Path(fn).suffix.lower()
            if ext in CODE_EXT:
                lang, is_doc = CODE_EXT[ext], False
            elif ext in DOC_EXT:
                lang, is_doc = DOC_EXT[ext], True
            else:
                continue
            full = Path(dirpath) / fn
            try:
                st = full.stat()
                if st.st_size > max_file_bytes or st.st_size == 0 or (ext != ".pdf" and _looks_binary(full)):
                    continue
                data = full.read_bytes()
            except OSError:
                continue
            text = _pdf_text(data) if ext == ".pdf" else data.decode("utf-8", errors="replace")
            if not text.strip():
                continue
            yield SourceFile(rel, lang, is_doc, text, hashlib.sha1(data).hexdigest(), st.st_mtime)