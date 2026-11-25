"""Counts questions per day"""
import sqlite3
from pathlib import Path


class UsageCounter:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS usage(day TEXT PRIMARY KEY, n INTEGER NOT NULL DEFAULT 0)")
        self.db.commit()

    def count(self, day: str) -> int:
        row = self.db.execute("SELECT n FROM usage WHERE day=?", (day,)).fetchone()
        return int(row[0]) if row else 0

    def bump(self, day: str) -> int:
        """Add one question to `day` and return the new total."""
        self.db.execute("INSERT INTO usage(day,n) VALUES(?,1) ON CONFLICT(day) DO UPDATE SET n=n+1", (day,))
        self.db.commit()
        return self.count(day)