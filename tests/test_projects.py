"""Tests for indexing the last three projects"""
import tempfile
from pathlib import Path

from rag.config import Config
from rag.pipeline import Engine
from rag.projects import ProjectStore, make_slug
from rag.usage import UsageCounter
from tests.fakes import FakeEmbedder


def make_project(folder: Path, word: str):
    folder.mkdir()
    (folder / "app.py").write_text(f"def {word}_handler():\n    return '{word}'\n")
    (folder / "README.md").write_text(f"# {word}\nThis project is all about {word}.\n")


def make_engine(tmp: Path) -> Engine:
    cfg = Config()
    cfg.index_dir, cfg.repos_dir, cfg.embed_model, cfg.max_commits = tmp / "idx", tmp / "repos", "fake", 0
    return Engine(cfg, FakeEmbedder(), ProjectStore(cfg.index_dir, 3), UsageCounter(cfg.index_dir / "usage.db"))


def test_slug_is_stable_and_readable():
    slug1, name1 = make_slug("https://github.com/Owner/Repo")
    slug2, name2 = make_slug("https://github.com/Owner/Repo.git/")
    assert slug1 == slug2 and name1 == name2 == "Owner/Repo"
    assert slug1.startswith("owner-repo-")
    assert make_slug("git@github.com:Owner/Repo.git")[1] == "Owner/Repo"


def test_keeps_only_the_last_three_projects():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        engine = make_engine(tmp)
        for word in ["alpha", "bravo", "charlie", "delta"]:
            make_project(tmp / word, word)
            engine.index_repo(str(tmp / word))
        assert [p.name for p in engine.projects.list()] == ["delta", "charlie", "bravo"]
        alpha_slug, _ = make_slug(str(tmp / "alpha"))
        assert not engine.projects.path(alpha_slug).exists()          

        engine.index_repo(str(tmp / "bravo"))                          
        assert [p.name for p in engine.projects.list()] == ["bravo", "delta", "charlie"]

        # the list survives a restart
        again = ProjectStore(tmp / "idx", 3)
        assert [p.name for p in again.list()] == ["bravo", "delta", "charlie"]


def test_each_project_is_searched_separately():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        engine = make_engine(tmp)
        for word in ["alpha", "bravo"]:
            make_project(tmp / word, word)
            engine.index_repo(str(tmp / word))
        alpha, _ = make_slug(str(tmp / "alpha"))
        bravo, _ = make_slug(str(tmp / "bravo"))
        hits = engine.retrieve("alpha_handler", alpha, top_k=5)
        assert hits and all("bravo" not in h.chunk.content for h in hits)
        hits = engine.retrieve("bravo_handler", bravo, top_k=5)
        assert hits and all("alpha" not in h.chunk.content for h in hits)


def test_only_our_own_clones_are_pulled():
    import rag.pipeline as pipeline
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        engine = make_engine(tmp)
        pulled = []
        original = pipeline.ingest.fetch_remote_updates
        pipeline.ingest.fetch_remote_updates = lambda root: pulled.append(root)
        try:
            make_project(tmp / "my_own_folder", "alpha")                      
            engine.index_repo(str(tmp / "my_own_folder"))
            engine.has_new_commits(make_slug(str(tmp / "my_own_folder"))[0])
            assert pulled == []

            engine.cfg.repos_dir.mkdir(parents=True)
            make_project(engine.cfg.repos_dir / "a_clone", "bravo")           
            engine.index_repo(str(engine.cfg.repos_dir / "a_clone"))
            engine._last_fetch = 0.0
            engine.has_new_commits(make_slug(str(engine.cfg.repos_dir / "a_clone"))[0])
            assert len(pulled) == 1
        finally:
            pipeline.ingest.fetch_remote_updates = original


def test_blank_source_is_refused():
    with tempfile.TemporaryDirectory() as d:
        engine = make_engine(Path(d))
        refused = False
        try:
            engine.index_repo("   ")
        except ValueError:
            refused = True
        assert refused and engine.projects.list() == []


def test_backup_model_answers_when_the_main_model_is_busy():
    from types import SimpleNamespace
    import rag.llm as llm_module
    import rag.pipeline as pipeline

    class Fake:
        def __init__(self, name, mode):
            self.name, self.mode, self.calls = name, mode, 0

        def stream(self, msgs):
            self.calls += 1
            if self.mode == "busy":
                raise RuntimeError("503 UNAVAILABLE. high demand")
            if self.mode == "partial":
                yield SimpleNamespace(content="partial ")
                raise RuntimeError("503 UNAVAILABLE. high demand")
            yield SimpleNamespace(content=f"answer from {self.name}")

    made = {}

    def fake_get_llm(cfg, model=None):
        name = model or "main"
        made.setdefault(name, Fake(name, made.get("_main_mode", "busy") if name == "main" else "ok"))
        return made[name]

    original_get, original_sleep = pipeline.get_llm, llm_module.time.sleep
    pipeline.get_llm, llm_module.time.sleep = fake_get_llm, lambda seconds: None
    try:
        with tempfile.TemporaryDirectory() as d:
            engine = make_engine(Path(d))
            engine.cfg.llm_fallback_model = "backup"
            notes = []
            stream, _ = engine.answer_stream("q", [], None, on_fallback=notes.append)
            assert "".join(stream) == "answer from backup" and notes == ["backup"]
            assert made["main"].calls == 2                       

            engine.cfg.llm_fallback_model = ""                   
            made["main"].calls = 0
            stream, _ = engine.answer_stream("q", [], None)
            raised = False
            try:
                list(stream)
            except RuntimeError:
                raised = True
            assert raised and made["main"].calls == 4            

            engine.cfg.llm_fallback_model = "backup"             
            made["main"].mode, made["backup"].calls = "partial", 0
            pieces, raised = [], False
            stream, _ = engine.answer_stream("q", [], None)
            try:
                for piece in stream:
                    pieces.append(piece)
            except RuntimeError:
                raised = True
            assert pieces == ["partial "] and raised and made["backup"].calls == 0
    finally:
        pipeline.get_llm, llm_module.time.sleep = original_get, original_sleep


def test_usage_counter_is_shared():
    with tempfile.TemporaryDirectory() as d:
        counter = UsageCounter(Path(d) / "usage.db")
        assert counter.count("2026-01-01") == 0
        assert counter.bump("2026-01-01") == 1 and counter.bump("2026-01-01") == 2
        assert UsageCounter(Path(d) / "usage.db").count("2026-01-01") == 2   


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)