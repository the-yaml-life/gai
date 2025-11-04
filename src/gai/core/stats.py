"""
Stats tracking for gai usage

Supports both JSON (legacy) and database backends (SQLite/PostgreSQL)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class Stats:
    """Track gai usage statistics - supports JSON and DB backends"""

    def __init__(self, stats_file: Optional[Path] = None, use_db: bool = False, db_backend: str = "sqlite", db_config: Optional[Dict] = None):
        """
        Initialize stats tracking.

        Args:
            stats_file: Path to JSON stats file (for legacy mode)
            use_db: Whether to use database backend
            db_backend: "sqlite" or "postgresql"
            db_config: Database configuration (db_path for sqlite, connection_string for postgresql)
        """
        self.use_db = use_db

        if use_db:
            # Import and initialize DB backend
            from gai.core.db_storage import get_storage_backend

            db_config = db_config or {}
            self.storage = get_storage_backend(backend_type=db_backend, **db_config)
        else:
            # Use JSON file backend
            if stats_file is None:
                # Store in .gai_stats.json in gai directory
                stats_file = Path(__file__).parent.parent / ".gai_stats.json"

            self.stats_file = stats_file
            self.data = self._load()
            self.storage = None

    def _load(self) -> Dict[str, Any]:
        """Load stats from file"""
        if not self.stats_file.exists():
            return self._default_stats()

        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except:
            return self._default_stats()

    def _default_stats(self) -> Dict[str, Any]:
        """Default stats structure"""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "commands": {
                "commit": {"count": 0, "last_used": None},
                "diff": {"count": 0, "last_used": None},
                "review_merge": {"count": 0, "last_used": None},
            },
            "api": {
                "total_calls": 0,
                "total_tokens": 0,
                "by_model": {},
                "by_day": {},
            },
            "commits": {
                "total": 0,
                "auto": 0,
                "manual_edit": 0,
                "dry_run": 0,
            }
        }

    def _save(self):
        """Save stats to file"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def record_command(self, command: str):
        """Record command usage"""
        if self.use_db:
            # DB backend tracks commands through save_record calls
            # This is just a marker, actual recording happens in command execution
            pass
        else:
            # JSON backend
            if command not in self.data["commands"]:
                self.data["commands"][command] = {"count": 0, "last_used": None}

            self.data["commands"][command]["count"] += 1
            self.data["commands"][command]["last_used"] = datetime.now().isoformat()

            self._save()

    def record_api_call(
        self,
        model: str,
        tokens_used: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ):
        """Record API call"""
        if self.use_db:
            # DB backend: Store in context for later save_record call
            # This gets called from groq_client during API calls
            # The actual record is saved when the command completes
            if not hasattr(self, '_pending_record'):
                self._pending_record = {}

            self._pending_record.update({
                "model_used": model,
                "total_tokens": tokens_used,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            })
        else:
            # JSON backend
            # Total stats
            self.data["api"]["total_calls"] += 1
            self.data["api"]["total_tokens"] += tokens_used

            # By model
            if model not in self.data["api"]["by_model"]:
                self.data["api"]["by_model"][model] = {
                    "calls": 0,
                    "tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0
                }

            self.data["api"]["by_model"][model]["calls"] += 1
            self.data["api"]["by_model"][model]["tokens"] += tokens_used
            self.data["api"]["by_model"][model]["prompt_tokens"] += prompt_tokens
            self.data["api"]["by_model"][model]["completion_tokens"] += completion_tokens

            # By day
            today = datetime.now().strftime("%Y-%m-%d")
            if today not in self.data["api"]["by_day"]:
                self.data["api"]["by_day"][today] = {
                    "calls": 0,
                    "tokens": 0
                }

            self.data["api"]["by_day"][today]["calls"] += 1
            self.data["api"]["by_day"][today]["tokens"] += tokens_used

            self._save()

    def record_commit(self, auto: bool = False, dry_run: bool = False, edited: bool = False):
        """Record commit creation"""
        if self.use_db:
            # DB backend: Update pending record with commit info
            if not hasattr(self, '_pending_record'):
                self._pending_record = {}

            self._pending_record.update({
                "committed": not dry_run,
                "context_provided": "edited" if edited else ("auto" if auto else None)
            })
        else:
            # JSON backend
            self.data["commits"]["total"] += 1

            if dry_run:
                self.data["commits"]["dry_run"] += 1
            elif auto:
                self.data["commits"]["auto"] += 1
            elif edited:
                self.data["commits"]["manual_edit"] += 1

            self._save()

    def save_record(
        self,
        command_type: str,
        generated_output: str,
        repo_path: str = None,
        branch_name: str = None,
        commit_hash: str = None,
        diff_content: str = None,
        status_content: str = None,
        stats_content: str = None,
        commit_type: str = None,
        commit_scope: str = None,
        breaking_change: bool = False,
        success: bool = True,
        error_message: str = None
    ):
        """
        Save a complete record to database.
        Only used with DB backend.

        Args:
            command_type: "commit", "review_merge", or "diff"
            generated_output: The AI-generated output (commit message, analysis, etc.)
            repo_path: Path to git repository
            branch_name: Current branch name
            commit_hash: Commit hash (if applicable)
            diff_content: Git diff content
            status_content: Git status content
            stats_content: Git stats content
            commit_type: Type of commit (feat, fix, etc.)
            commit_scope: Scope of commit
            breaking_change: Whether this is a breaking change
            success: Whether the operation was successful
            error_message: Error message if not successful
        """
        if not self.use_db:
            # JSON backend doesn't use save_record
            return

        # Get pending record data (from record_api_call)
        pending = getattr(self, '_pending_record', {})

        # Build complete record
        record = {
            "timestamp": datetime.now().isoformat(),
            "command_type": command_type,
            "repo_path": repo_path,
            "branch_name": branch_name,
            "commit_hash": commit_hash,
            "model_used": pending.get("model_used"),
            "prompt_tokens": pending.get("prompt_tokens"),
            "completion_tokens": pending.get("completion_tokens"),
            "total_tokens": pending.get("total_tokens"),
            "diff_content": diff_content,
            "status_content": status_content,
            "stats_content": stats_content,
            "generated_output": generated_output,
            "committed": pending.get("committed", False),
            "commit_type": commit_type,
            "commit_scope": commit_scope,
            "context_provided": pending.get("context_provided"),
            "breaking_change": breaking_change,
            "success": success,
            "error_message": error_message
        }

        # Save to database
        self.storage.save_record(record)

        # Clear pending record
        self._pending_record = {}

    def get_summary(self) -> Dict[str, Any]:
        """Get stats summary"""
        if self.use_db:
            # DB backend: Get stats from database
            return self.storage.get_stats()
        else:
            # JSON backend
            today = datetime.now().strftime("%Y-%m-%d")
            today_stats = self.data["api"]["by_day"].get(today, {"calls": 0, "tokens": 0})

            # Calculate tokens this week
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            week_tokens = sum(
                day_data["tokens"]
                for day, day_data in self.data["api"]["by_day"].items()
                if day >= week_ago
            )

            return {
                "commands": {
                    "total": sum(cmd["count"] for cmd in self.data["commands"].values()),
                    "breakdown": self.data["commands"]
                },
                "api": {
                    "total_calls": self.data["api"]["total_calls"],
                    "total_tokens": self.data["api"]["total_tokens"],
                    "today_calls": today_stats["calls"],
                    "today_tokens": today_stats["tokens"],
                    "week_tokens": week_tokens,
                    "by_model": self.data["api"]["by_model"]
                },
                "commits": self.data["commits"],
                "created_at": self.data.get("created_at"),
            }

    def get_today_usage(self) -> Dict[str, int]:
        """Get today's usage for rate limit tracking"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_stats = self.data["api"]["by_day"].get(today, {"calls": 0, "tokens": 0})

        return {
            "calls": today_stats["calls"],
            "tokens": today_stats["tokens"]
        }

    def reset(self):
        """Reset all stats"""
        if self.use_db:
            # DB backend: Would need to truncate tables
            # For now, we don't support reset for DB backend
            raise NotImplementedError("Reset not supported for database backend")
        else:
            # JSON backend
            self.data = self._default_stats()
            self._save()


# Global stats instance
_stats: Optional[Stats] = None


def get_stats(config: Optional[Any] = None) -> Stats:
    """
    Get global stats instance.

    Args:
        config: Optional config object to initialize stats with DB backend
    """
    global _stats
    if _stats is None:
        if config is None:
            # Default: JSON backend
            _stats = Stats()
        else:
            # Check if DB backend is configured
            use_db = config.get('stats.use_db', False)

            if use_db:
                db_backend = config.get('stats.backend', 'sqlite')
                db_config = {}

                if db_backend == 'sqlite':
                    db_config['db_path'] = config.get('stats.db_path', None)
                elif db_backend == 'postgresql':
                    db_config['connection_string'] = config.get('stats.connection_string')

                _stats = Stats(use_db=True, db_backend=db_backend, db_config=db_config)
            else:
                # JSON backend
                _stats = Stats()

    return _stats
