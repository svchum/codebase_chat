"""finds out which part of the app is failing or slow. Every step has a time limit : python scripts/check_setup.py """
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_config  
from rag.llm import get_llm, stream_answer  
from rag.pipeline import Engine  

GOOGLE_API = "https://generativelanguage.googleapis.com/v1beta"
TINY_QUESTION = {"contents": [{"parts": [{"text": "Reply with the single word: ok"}]}]}
SKIP_WORDS = ("image", "tts", "live", "audio", "embedding", "robotics", "computer-use", "native", "vision")


def step(title, action):
    """Run one check, print how long it took, and show the full error if it fails."""
    print(f"\n== {title}")
    started = time.perf_counter()
    try:
        result = action()
        print(f"   OK ({time.perf_counter() - started:.1f}s) {result or ''}")
        return True
    except Exception: 
        print(f"   FAILED after {time.perf_counter() - started:.1f}s")
        traceback.print_exc()
        return False


def google_request(path, key, payload=None, timeout=30):
    """Talk to Google's API directly (no AI libraries), so we can tell a network problem from a library problem."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(f"{GOOGLE_API}/{path}", data=data,
                                     headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Google answered HTTP {error.code}: {error.read().decode(errors='replace')[:400]}") from None


def main() -> int:
    cfg = load_config()
    models = {"gemini": cfg.gemini_model, "openai": cfg.openai_model,
              "anthropic": cfg.anthropic_model, "claude": cfg.anthropic_model}
    model = models.get(cfg.llm_provider, "?")
    print("CodebaseChat setup check")
    print(f"   mode: {'demo' if cfg.demo_mode else 'personal'} | AI: {cfg.llm_provider} / {model}"
          + (f" | backup: {cfg.llm_fallback_model}" if cfg.llm_fallback_model else ""))
    print(f"   index folder: {cfg.index_dir}")

    engine, catalog, results = None, [], {}

    def open_engine():
        nonlocal engine
        engine = Engine.create(cfg)
        projects = engine.projects.list()
        if not projects:
            raise RuntimeError("no projects are indexed yet. Run: python scripts/reindex.py --repo <url or folder>")
        return "projects: " + ", ".join(p.name for p in projects)

    ok = step("1. Open the index folder and list projects", open_engine)
    slug = engine.projects.list()[0].slug if ok else None
    ok = ok and step("2. Load the search model and embed a question", lambda: engine.embedder.embed_query("hello") is not None and "")
    results["search"] = ok and step("3. Search the newest project",
                                    lambda: f"{len(engine.retrieve('overview of this project', slug))} snippets found")

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if cfg.llm_provider == "gemini":
        def list_models():
            if not key:
                raise RuntimeError("GOOGLE_API_KEY is not set (check your .env file)")
            for item in google_request("models?pageSize=200", key)["models"]:
                if "generateContent" in item.get("supportedGenerationMethods", []):
                    catalog.append(item["name"].removeprefix("models/"))
            if model not in catalog:
                flash = sorted(n for n in catalog if "flash" in n)
                raise RuntimeError(f"your model '{model}' is not available to your key. Flash models that are: {flash[:12]}")
            return f"'{model}' is available ({len(catalog)} models can generate text)"

        results["google"] = step("4. Reach Google with your key and check the model name (30s limit)", list_models)

        def direct_call():
            reply = google_request(f"models/{model}:generateContent", key, TINY_QUESTION, timeout=45)
            return "reply: " + repr(reply["candidates"][0]["content"]["parts"][0].get("text", "").strip()[:40])

        results["direct"] = results["google"] and step("5. Ask your model directly, no AI libraries (45s limit)", direct_call)

        if results["google"] and not results["direct"]:
            def probe_other_models():
                candidates = [n for n in catalog if "flash" in n and n != model and not any(w in n for w in SKIP_WORDS)]
                candidates.sort(key=lambda n: ("preview" in n or "exp" in n, n))  
                working = []
                for name in candidates[:6]:
                    began = time.perf_counter()
                    try:
                        google_request(f"models/{name}:generateContent", key, TINY_QUESTION, timeout=20)
                        took = time.perf_counter() - began
                        working.append((took, name))
                        outcome = f"OK in {took:.1f}s"
                    except RuntimeError as error:
                        outcome = str(error).split(":", 1)[0]         
                    except Exception as error:  
                        outcome = type(error).__name__                
                    print(f"      {name}: {outcome}")
                if not working:
                    raise RuntimeError("none of the other Flash models answered either. This looks like a wider Google problem; try again later.")
                best = min(working)[1]
                return f"suggestion: put GEMINI_MODEL={best} in your .env (and optionally LLM_FALLBACK_MODEL=<another working one>)"

            results["others"] = step("5b. Your model failed. Trying your other Flash models (20s limit each)", probe_other_models)

    if results.get("direct") is False:
        print("\n== 6. Ask through the app's own AI code\n   skipped: Google itself is failing for this model (see step 5)")
        results["app_ai"] = None
    else:
        def ask_app_ai():
            llm = get_llm(cfg)
            reply = "".join(stream_answer(llm, [{"role": "user", "content": "Reply with the single word: ok"}],
                                          retries=0, wait_seconds=45))
            return f"reply: {reply.strip()[:40]!r}"

        results["app_ai"] = step("6. Ask through the app's own AI code (45s limit)", ask_app_ai)

    def show(value):
        return "skipped" if value is None else "OK" if value else "FAILED"

    print("\n== Summary")
    print("   search:        ", show(results["search"]))
    if cfg.llm_provider == "gemini":
        print("   Google/model:  ", show(results["google"]), "" if results["google"] else "(network, key, or model name)")
        print("   your model:    ", show(results["direct"]))
        if "others" in results:
            print("   other models:  ", show(results["others"]))
    print("   app AI code:   ", show(results["app_ai"]))
    if results.get("direct") is False:
        print("\n   Google answered with an error for your model, so this is not your app. "
              "Switch to a model that works (see 5b above).")
    elif cfg.llm_provider == "gemini" and results.get("direct") and results["app_ai"] is False:
        print("\n   Google works when called directly, so the problem is in the AI library, not your network or key.")
    return 0 if all(v in (True, None) for v in results.values()) and results.get("direct") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())