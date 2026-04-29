import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path


class FTS5Store:
    def __init__(self, db_path: str, retention_days: int = 90):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS history_fts
                USING fts5(content, content_rowid='id')
            """)
            # Triggers to keep FTS in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS history_ai AFTER INSERT ON history
                BEGIN
                    INSERT INTO history_fts(rowid, content) VALUES (new.id, new.content);
                END
            """)
            conn.commit()

    def insert(self, role: str, content: str):
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "INSERT INTO history (role, content, created_at) VALUES (?, ?, ?)",
                (role, content, time.time()),
            )
            conn.commit()
            return cursor.lastrowid

    def search(self, query: str, limit: int = 3) -> list[dict]:
        with sqlite3.connect(str(self.db_path)) as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT h.id, h.role, h.content, h.created_at
                    FROM history h
                    JOIN history_fts fts ON h.id = fts.rowid
                    WHERE history_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            return [
                {"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]}
                for r in rows
            ]

    def get_recent(self, limit: int = 10) -> list[dict]:
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT id, role, content, created_at FROM history ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]}
            for r in rows
        ]

    def cleanup(self):
        if self.retention_days <= 0:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("DELETE FROM history")
                conn.execute("DELETE FROM history_fts")
                conn.commit()
            return
        cutoff = (datetime.now() - timedelta(days=self.retention_days)).timestamp()
        with sqlite3.connect(str(self.db_path)) as conn:
            old_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM history WHERE created_at < ?", (cutoff,)
                ).fetchall()
            ]
            if old_ids:
                conn.executemany("DELETE FROM history WHERE id = ?", [(i,) for i in old_ids])
                conn.executemany("DELETE FROM history_fts WHERE rowid = ?", [(i,) for i in old_ids])
            conn.commit()
