#!/usr/bin/env python
"""index a repository. Used locally and on GitHub """
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_config  
from rag.pipeline import Engine  


def main() -> int:
    ap = argparse.ArgumentParser(description="Index a repo for Chat-with-your-Codebase")
    ap.add_argument("--repo", required=True, help="Local path or Git URL")
    ap.add_argument("--index-dir", help="Override INDEX_DIR")
    ap.add_argument("--full", action="store_true", help="Force a full rebuild")
    args = ap.parse_args()

    cfg = load_config()
    if args.index_dir:
        cfg.index_dir = Path(args.index_dir)
    engine = Engine.create(cfg)

    last = {"stage": None}

    def progress(stage, done, total):
        if stage != last["stage"] or (total and done == total):
            print(f"  [{stage}] {done}" + (f"/{total}" if total else ""), flush=True)
            last["stage"] = stage

    print(f"Indexing {args.repo} -> {cfg.index_dir}")
    stats = engine.index_repo(args.repo, progress, full=args.full)
    print("Done:", stats.summary())
    print("Projects kept (newest first):", ", ".join(p.name for p in engine.projects.list()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())