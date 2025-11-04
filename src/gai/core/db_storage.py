"""
Storage backends for gai

Supports SQLite (default) and PostgreSQL (optional)
Adapted from git-commit-ai to support multiple command types (commit, review-merge, diff)
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
from abc import ABC, abstractmethod


# ============================================================================
# Schema Definition
# ============================================================================

SCHEMA_VERSION = 1

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS gai_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    command_type TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    branch_name TEXT,
    commit_hash TEXT,
    model_used TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    diff_content TEXT,
    status_content TEXT,
    stats_content TEXT,
    generated_output TEXT NOT NULL,
    committed BOOLEAN DEFAULT 0,
    commit_type TEXT,
    commit_scope TEXT,
    context_provided TEXT,
    breaking_change BOOLEAN DEFAULT 0,
    success BOOLEAN DEFAULT 1,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON gai_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_command_type ON gai_records(command_type);
CREATE INDEX IF NOT EXISTS idx_repo_path ON gai_records(repo_path);
CREATE INDEX IF NOT EXISTS idx_model_used ON gai_records(model_used);
CREATE INDEX IF NOT EXISTS idx_commit_type ON gai_records(commit_type);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS gai_records (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    command_type TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    branch_name TEXT,
    commit_hash TEXT,
    model_used TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    diff_content TEXT,
    status_content TEXT,
    stats_content TEXT,
    generated_output TEXT NOT NULL,
    committed BOOLEAN DEFAULT FALSE,
    commit_type TEXT,
    commit_scope TEXT,
    context_provided TEXT,
    breaking_change BOOLEAN DEFAULT FALSE,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON gai_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_command_type ON gai_records(command_type);
CREATE INDEX IF NOT EXISTS idx_repo_path ON gai_records(repo_path);
CREATE INDEX IF NOT EXISTS idx_model_used ON gai_records(model_used);
CREATE INDEX IF NOT EXISTS idx_commit_type ON gai_records(commit_type);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


# ============================================================================
# Abstract Storage Backend
# ============================================================================

class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def initialize(self):
        """Initialize database and create schema if needed."""
        pass

    @abstractmethod
    def save_record(self, data: Dict[str, Any]) -> int:
        """Save a gai record. Returns the record ID."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        pass

    @abstractmethod
    def get_history(self, limit: int = 10, repo_path: Optional[str] = None, command_type: Optional[str] = None) -> List[Dict]:
        """Get command history."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search records by output content."""
        pass

    @abstractmethod
    def close(self):
        """Close database connection."""
        pass


# ============================================================================
# SQLite Backend
# ============================================================================

class SQLiteStorage(StorageBackend):
    """SQLite storage backend."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".gai.db")

        self.db_path = db_path
        self.conn = None

    def initialize(self):
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Create schema
        self.conn.executescript(SQLITE_SCHEMA)

        # Store schema version
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION))
        )
        self.conn.commit()

    def save_record(self, data: Dict[str, Any]) -> int:
        """Save a gai record."""
        cursor = self.conn.execute("""
            INSERT INTO gai_records (
                timestamp, command_type, repo_path, branch_name, commit_hash,
                model_used, prompt_tokens, completion_tokens, total_tokens,
                diff_content, status_content, stats_content,
                generated_output, committed, commit_type, commit_scope,
                context_provided, breaking_change, success, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("timestamp", datetime.now().isoformat()),
            data.get("command_type"),
            data.get("repo_path"),
            data.get("branch_name"),
            data.get("commit_hash"),
            data.get("model_used"),
            data.get("prompt_tokens"),
            data.get("completion_tokens"),
            data.get("total_tokens"),
            data.get("diff_content"),
            data.get("status_content"),
            data.get("stats_content"),
            data.get("generated_output"),
            data.get("committed", False),
            data.get("commit_type"),
            data.get("commit_scope"),
            data.get("context_provided"),
            data.get("breaking_change", False),
            data.get("success", True),
            data.get("error_message")
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        # Total records
        total = self.conn.execute("SELECT COUNT(*) FROM gai_records").fetchone()[0]

        # Total tokens
        total_tokens = self.conn.execute(
            "SELECT SUM(total_tokens) FROM gai_records WHERE total_tokens IS NOT NULL"
        ).fetchone()[0] or 0

        # Today's stats
        today = datetime.now().date().isoformat()
        today_result = self.conn.execute("""
            SELECT COUNT(*) as calls, SUM(total_tokens) as tokens
            FROM gai_records
            WHERE DATE(timestamp) = ? AND total_tokens IS NOT NULL
        """, (today,)).fetchone()

        today_calls = today_result[0]
        today_tokens = today_result[1] or 0

        # Week's tokens
        week_ago = (datetime.now().date() - __import__('datetime').timedelta(days=7)).isoformat()
        week_tokens = self.conn.execute("""
            SELECT SUM(total_tokens)
            FROM gai_records
            WHERE DATE(timestamp) >= ? AND total_tokens IS NOT NULL
        """, (week_ago,)).fetchone()[0] or 0

        # Stats by command type
        command_stats = {}
        for row in self.conn.execute("""
            SELECT command_type, COUNT(*) as count, MAX(timestamp) as last_used
            FROM gai_records
            GROUP BY command_type
        """).fetchall():
            command_stats[row[0]] = {
                "count": row[1],
                "last_used": row[2]
            }

        # Stats by model
        model_stats = {}
        for row in self.conn.execute("""
            SELECT model_used, COUNT(*) as calls, SUM(total_tokens) as tokens,
                   SUM(prompt_tokens) as prompt_tokens, SUM(completion_tokens) as completion_tokens
            FROM gai_records
            WHERE total_tokens IS NOT NULL
            GROUP BY model_used
        """).fetchall():
            model_stats[row[0]] = {
                "calls": row[1],
                "tokens": row[2] or 0,
                "prompt_tokens": row[3] or 0,
                "completion_tokens": row[4] or 0
            }

        # Commit-specific stats
        commit_total = self.conn.execute(
            "SELECT COUNT(*) FROM gai_records WHERE command_type = 'commit'"
        ).fetchone()[0]

        commit_auto = self.conn.execute(
            "SELECT COUNT(*) FROM gai_records WHERE command_type = 'commit' AND committed = 1 AND context_provided IS NULL"
        ).fetchone()[0]

        commit_manual = self.conn.execute(
            "SELECT COUNT(*) FROM gai_records WHERE command_type = 'commit' AND committed = 1 AND context_provided IS NOT NULL"
        ).fetchone()[0]

        commit_dry_run = self.conn.execute(
            "SELECT COUNT(*) FROM gai_records WHERE command_type = 'commit' AND committed = 0"
        ).fetchone()[0]

        # Last used
        last_used = self.conn.execute(
            "SELECT MAX(timestamp) FROM gai_records"
        ).fetchone()[0]

        return {
            "version": "1.0",
            "created_at": last_used.split("T")[0] if last_used else None,
            "commands": {
                "total": total,
                "breakdown": command_stats
            },
            "api": {
                "total_calls": total,
                "total_tokens": int(total_tokens),
                "today_calls": today_calls,
                "today_tokens": int(today_tokens),
                "week_tokens": int(week_tokens),
                "by_model": model_stats
            },
            "commits": {
                "total": commit_total,
                "auto": commit_auto,
                "manual_edit": commit_manual,
                "dry_run": commit_dry_run
            }
        }

    def get_history(self, limit: int = 10, repo_path: Optional[str] = None, command_type: Optional[str] = None) -> List[Dict]:
        """Get command history."""
        query = "SELECT * FROM gai_records WHERE 1=1"
        params = []

        if repo_path:
            query += " AND repo_path = ?"
            params.append(repo_path)

        if command_type:
            query += " AND command_type = ?"
            params.append(command_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search records by output content."""
        rows = self.conn.execute("""
            SELECT * FROM gai_records
            WHERE generated_output LIKE ? OR diff_content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit)).fetchall()

        return [dict(row) for row in rows]

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# ============================================================================
# PostgreSQL Backend
# ============================================================================

class PostgreSQLStorage(StorageBackend):
    """PostgreSQL storage backend."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None

        # Check if psycopg2 is available
        try:
            import psycopg2
            import psycopg2.extras
            self.psycopg2 = psycopg2
        except ImportError:
            raise ImportError(
                "PostgreSQL backend requires psycopg2. Install with: pip install psycopg2-binary"
            )

    def initialize(self):
        """Initialize PostgreSQL database."""
        self.conn = self.psycopg2.connect(self.connection_string)
        self.conn.autocommit = True

        # Create schema
        with self.conn.cursor() as cursor:
            cursor.execute(POSTGRES_SCHEMA)

            # Store schema version
            cursor.execute(
                "INSERT INTO metadata (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                ("schema_version", str(SCHEMA_VERSION))
            )

    def save_record(self, data: Dict[str, Any]) -> int:
        """Save a gai record."""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO gai_records (
                    timestamp, command_type, repo_path, branch_name, commit_hash,
                    model_used, prompt_tokens, completion_tokens, total_tokens,
                    diff_content, status_content, stats_content,
                    generated_output, committed, commit_type, commit_scope,
                    context_provided, breaking_change, success, error_message
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                data.get("timestamp", datetime.now()),
                data.get("command_type"),
                data.get("repo_path"),
                data.get("branch_name"),
                data.get("commit_hash"),
                data.get("model_used"),
                data.get("prompt_tokens"),
                data.get("completion_tokens"),
                data.get("total_tokens"),
                data.get("diff_content"),
                data.get("status_content"),
                data.get("stats_content"),
                data.get("generated_output"),
                data.get("committed", False),
                data.get("commit_type"),
                data.get("commit_scope"),
                data.get("context_provided"),
                data.get("breaking_change", False),
                data.get("success", True),
                data.get("error_message")
            ))
            return cursor.fetchone()[0]

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        with self.conn.cursor(cursor_factory=self.psycopg2.extras.RealDictCursor) as cursor:
            # Total records
            cursor.execute("SELECT COUNT(*) as total FROM gai_records")
            total = cursor.fetchone()["total"]

            # Total tokens
            cursor.execute("SELECT SUM(total_tokens) as total FROM gai_records WHERE total_tokens IS NOT NULL")
            total_tokens = cursor.fetchone()["total"] or 0

            # Today's stats
            cursor.execute("""
                SELECT COUNT(*) as calls, SUM(total_tokens) as tokens
                FROM gai_records
                WHERE DATE(timestamp) = CURRENT_DATE AND total_tokens IS NOT NULL
            """)
            today_result = cursor.fetchone()
            today_calls = today_result["calls"]
            today_tokens = today_result["tokens"] or 0

            # Week's tokens
            cursor.execute("""
                SELECT SUM(total_tokens) as total
                FROM gai_records
                WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days' AND total_tokens IS NOT NULL
            """)
            week_tokens = cursor.fetchone()["total"] or 0

            # Stats by command type
            cursor.execute("""
                SELECT command_type, COUNT(*) as count, MAX(timestamp) as last_used
                FROM gai_records
                GROUP BY command_type
            """)
            command_stats = {
                row["command_type"]: {
                    "count": row["count"],
                    "last_used": row["last_used"].isoformat() if row["last_used"] else None
                }
                for row in cursor.fetchall()
            }

            # Stats by model
            cursor.execute("""
                SELECT model_used, COUNT(*) as calls, SUM(total_tokens) as tokens,
                       SUM(prompt_tokens) as prompt_tokens, SUM(completion_tokens) as completion_tokens
                FROM gai_records
                WHERE total_tokens IS NOT NULL
                GROUP BY model_used
            """)
            model_stats = {
                row["model_used"]: {
                    "calls": row["calls"],
                    "tokens": row["tokens"] or 0,
                    "prompt_tokens": row["prompt_tokens"] or 0,
                    "completion_tokens": row["completion_tokens"] or 0
                }
                for row in cursor.fetchall()
            }

            # Commit-specific stats
            cursor.execute("SELECT COUNT(*) as total FROM gai_records WHERE command_type = 'commit'")
            commit_total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) as total FROM gai_records WHERE command_type = 'commit' AND committed = TRUE AND context_provided IS NULL")
            commit_auto = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) as total FROM gai_records WHERE command_type = 'commit' AND committed = TRUE AND context_provided IS NOT NULL")
            commit_manual = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) as total FROM gai_records WHERE command_type = 'commit' AND committed = FALSE")
            commit_dry_run = cursor.fetchone()["total"]

            # Last used
            cursor.execute("SELECT MAX(timestamp) as last_used FROM gai_records")
            last_used = cursor.fetchone()["last_used"]

            return {
                "version": "1.0",
                "created_at": last_used.date().isoformat() if last_used else None,
                "commands": {
                    "total": total,
                    "breakdown": command_stats
                },
                "api": {
                    "total_calls": total,
                    "total_tokens": int(total_tokens),
                    "today_calls": today_calls,
                    "today_tokens": int(today_tokens),
                    "week_tokens": int(week_tokens),
                    "by_model": model_stats
                },
                "commits": {
                    "total": commit_total,
                    "auto": commit_auto,
                    "manual_edit": commit_manual,
                    "dry_run": commit_dry_run
                }
            }

    def get_history(self, limit: int = 10, repo_path: Optional[str] = None, command_type: Optional[str] = None) -> List[Dict]:
        """Get command history."""
        with self.conn.cursor(cursor_factory=self.psycopg2.extras.RealDictCursor) as cursor:
            query = "SELECT * FROM gai_records WHERE TRUE"
            params = []

            if repo_path:
                query += " AND repo_path = %s"
                params.append(repo_path)

            if command_type:
                query += " AND command_type = %s"
                params.append(command_type)

            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search records by output content."""
        with self.conn.cursor(cursor_factory=self.psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT * FROM gai_records
                WHERE generated_output ILIKE %s OR diff_content ILIKE %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (f"%{query}%", f"%{query}%", limit))

            return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# ============================================================================
# Storage Factory
# ============================================================================

def get_storage_backend(backend_type: str = "sqlite", **kwargs) -> StorageBackend:
    """
    Factory function to get the appropriate storage backend.

    Args:
        backend_type: "sqlite" or "postgresql"
        **kwargs: Backend-specific arguments
            - For sqlite: db_path (optional)
            - For postgresql: connection_string (required)
    """
    if backend_type == "sqlite":
        storage = SQLiteStorage(db_path=kwargs.get("db_path"))
    elif backend_type == "postgresql":
        if "connection_string" not in kwargs:
            raise ValueError("PostgreSQL backend requires 'connection_string' parameter")
        storage = PostgreSQLStorage(connection_string=kwargs["connection_string"])
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    storage.initialize()
    return storage
