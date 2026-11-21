"""Run with pytest"""
import os
import tempfile
from pathlib import Path

from rag.config import Config
from rag.index import Filters, Index, fts_query, split_identifier
from rag.parse import chunk_file, window_split
from rag.prompt import build_messages, cited_indexes
from rag.retrieve import Retriever, extract_identifiers
from tests.fakes import FakeEmbedder

PY = '''import os

def authenticate_user(username, password):
    """Check credentials against the user store."""
    return username == "admin"

class ApiRouter:
    def handle_request(self, request):
        return process(request)

    def process_api_request(self, request):
        return {"ok": True}
'''


def make_repo(tmp: Path):
    (tmp / "auth.py").write_text(PY)
    (tmp / "README.md").write_text("# Project\nIntro text\n## Install\nrun pip install\n")
    (tmp / "index.html").write_text("<h1>Interests</h1><p>Chess and hiking. Studied at Example University.</p>")
    (tmp / "ignored.py").write_text("def secret(): pass\n")
    (tmp / ".gitignore").write_text("ignored.py\nbuild/\n")
    (tmp / "node_modules").mkdir()
    (tmp / "node_modules" / "x.js").write_text("function nope(){}")
    (tmp / "blob.py").write_bytes(b"\x00\x01\x02binary")


def cfg_for(tmp: Path) -> Config:
    c = Config()
    c.index_dir = tmp / "idx"
    c.embed_model = "fake"
    c.max_commits = 0
    return c


def test_window_split_and_markdown():
    lines = [f"line {i}" for i in range(100)]
    parts = window_split(lines, 0, 200, 2)
    assert parts[0][0] == 1 and parts[-1][1] == 100
    assert all(len(t) <= 200 + 10 for _, _, t in parts)
    md = chunk_file("r.md", "markdown", "# A\nx\n## B\ny\n", True)
    assert [c.name for c in md] == ["A", "B"]


def test_chunking_covers_the_whole_file():
    last_code_line = len(PY.rstrip("\n").split("\n"))

    chunks = chunk_file("a.py", "python", PY, False, max_chars=120, overlap=1)
    assert chunks and chunks[0].start_line == 1 and chunks[-1].end_line >= last_code_line

    window = chunk_file("a.rb", "ruby", PY, False, max_chars=120, overlap=1)
    assert window and all(c.kind == "window" for c in window) and window[-1].end_line >= last_code_line
    assert chunk_file("e.py", "python", "   \n", False) == []


def test_identifiers_and_fts():
    assert "authenticate_user" in extract_identifiers("where is `authenticate_user` called")
    assert "ApiRouter" in extract_identifiers("what does ApiRouter do")
    assert "handleRequest" in extract_identifiers("explain handleRequest()")
    assert split_identifier("getUserAuth_token") == "get User Auth token"
    assert '"auth"' in fts_query("Where is user authentication handled?") or '"authent' in fts_query("authentication")


def test_index_sync_incremental_and_retrieval():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        repo = tmp / "repo"
        repo.mkdir()
        make_repo(repo)
        cfg, emb = cfg_for(tmp), FakeEmbedder()
        idx = Index(cfg.index_dir)

        s1 = idx.sync(repo, emb, cfg)
        assert s1.files_added == 3 and s1.chunks_added >= 2        
        paths = {r[0] for r in idx.db.execute("SELECT DISTINCT file_path FROM chunks")}
        assert paths == {"auth.py", "README.md", "index.html"}, paths            
        assert idx.is_consistent()

        s2 = idx.sync(repo, emb, cfg)                           
        assert s2.files_unchanged == 3 and s2.chunks_added == 0

        (repo / "auth.py").write_text(PY + "\ndef brand_new_feature():\n    return 42\n")
        (repo / "extra.py").write_text("def other():\n    pass\n")
        s3 = idx.sync(repo, emb, cfg)
        assert s3.files_updated == 1 and s3.files_added == 1 and s3.files_unchanged == 2
        assert idx.is_consistent()

        (repo / "extra.py").unlink()
        s4 = idx.sync(repo, emb, cfg)
        assert s4.files_removed == 1 and idx.is_consistent()

        r = Retriever(idx, emb)
        hits = r.retrieve("where is brand_new_feature defined", k=5)
        assert hits and "brand_new_feature" in hits[0].chunk.content
        assert any(h.chunk.file_path == "auth.py" for h in hits)

        only_md = r.retrieve("install", k=5, filters=Filters(languages=["markdown"]))
        assert only_md and all(h.chunk.language == "markdown" for h in only_md)
        none = r.retrieve("install", k=5, filters=Filters(path_contains="does-not-exist"))
        assert none == []

        exp = r.expand(hits[:1])
        assert len(exp) >= 1

        idx2 = Index(cfg.index_dir)
        assert idx2.chunk_count() == idx.chunk_count() and idx2.vs.count == idx.vs.count

        cfg.embed_model = "other-model"
        s5 = idx2.sync(repo, emb, cfg)
        assert s5.rebuilt and idx2.is_consistent()


def test_prompt_building():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        repo = tmp / "repo"
        repo.mkdir()
        make_repo(repo)
        cfg, emb = cfg_for(tmp), FakeEmbedder()
        idx = Index(cfg.index_dir)
        idx.sync(repo, emb, cfg)
        hits = Retriever(idx, emb).retrieve("authenticate user", k=3)
        msgs, used = build_messages("How is auth done?", hits, history=[{"role": "user", "content": "hi"}])
        assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"
        assert "[1]" in msgs[-1]["content"] and "How is auth done?" in msgs[-1]["content"]
        assert used
    assert cited_indexes("See [2] and [1][2], not [9].", 3) == [2, 1]


def test_html_is_searchable():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        repo = tmp / "repo"
        repo.mkdir()
        make_repo(repo)
        cfg, emb = cfg_for(tmp), FakeEmbedder()
        idx = Index(cfg.index_dir)
        idx.sync(repo, emb, cfg)
        hits = Retriever(idx, emb).retrieve("what university and interests", k=3)
        assert any(h.chunk.file_path == "index.html" for h in hits)


def test_embedder_loads_the_model_only_once_even_with_several_threads():
    import sys
    import threading
    import time
    import types

    created = []

    class FakeModel:
        max_seq_length = 512

        def __init__(self, name):
            time.sleep(0.2)                       
            created.append(name)

        def get_embedding_dimension(self):
            return 4

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeModel
    saved = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = fake_module
    try:
        from rag.embed import Embedder
        embedder = Embedder("some-model")
        threads = [threading.Thread(target=embedder._load) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert created == ["some-model"] and embedder.dim == 4
    finally:
        if saved is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = saved


def test_llm_timeout_is_off_unless_set():
    import os
    from rag.config import Config
    saved = os.environ.pop("LLM_TIMEOUT_SECONDS", None)
    try:
        assert Config().llm_timeout is None
        os.environ["LLM_TIMEOUT_SECONDS"] = "30"
        assert Config().llm_timeout == 30.0
    finally:
        os.environ.pop("LLM_TIMEOUT_SECONDS", None)
        if saved is not None:
            os.environ["LLM_TIMEOUT_SECONDS"] = saved


def test_llm_gives_up_when_the_connection_stalls():
    import threading
    import time
    from types import SimpleNamespace
    from rag.llm import LlmTimeout, friendly_error, stream_answer

    class Stalls:
        def __init__(self, speak_first=False):
            self.speak_first, self.calls = speak_first, 0

        def stream(self, msgs):
            self.calls += 1
            if self.speak_first:
                yield SimpleNamespace(content="partial ")
            threading.Event().wait(2)              
            yield SimpleNamespace(content="never seen")

    msgs = [{"role": "user", "content": "hi"}]
    llm, began, raised = Stalls(), time.time(), None
    try:
        list(stream_answer(llm, msgs, wait=0, wait_seconds=0.2))
    except LlmTimeout as error:
        raised = error
    assert raised is not None and llm.calls == 2 and time.time() - began < 1.5   
    assert "too long" in friendly_error(raised)

    mid, pieces = Stalls(speak_first=True), []
    try:
        for piece in stream_answer(mid, msgs, wait=0, wait_seconds=0.2):
            pieces.append(piece)
    except LlmTimeout:
        pass
    assert pieces == ["partial "] and mid.calls == 1                             


def test_llm_retries_busy_errors_and_friendly_messages():
    from types import SimpleNamespace
    from rag.llm import friendly_error, stream_answer

    class Flaky:
        def __init__(self, fails):
            self.fails, self.calls = fails, 0

        def stream(self, msgs):
            self.calls += 1
            if self.calls <= self.fails:
                raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
            yield SimpleNamespace(content="hello ")
            yield SimpleNamespace(content="world")

    msgs = [{"role": "user", "content": "hi"}]
    seen = []
    llm = Flaky(2)
    out = "".join(stream_answer(llm, msgs, wait=0, on_retry=lambda a, t: seen.append(a)))
    assert out == "hello world" and llm.calls == 3 and seen == [1, 2]

    gave_up = False
    try:
        list(stream_answer(Flaky(10), msgs, retries=2, wait=0))
    except RuntimeError:
        gave_up = True
    assert gave_up

    assert "busy" in friendly_error(RuntimeError("503 UNAVAILABLE"))
    assert "limit" in friendly_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert friendly_error(ValueError("boom")) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)