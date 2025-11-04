"""
Configuration management for gai
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Manages gai configuration from .env and .gai.yaml with cascading search"""

    def __init__(self, config_path: Optional[Path] = None):
        # Find config file using cascading search
        if config_path is None:
            config_path = self._find_config()

        self.config_path = config_path

        # Load .env files with cascading (global + local override)
        self._load_env_cascade()

        # Load YAML config
        self.data = self._load_yaml()

        # Set default DB path relative to config location if not specified
        self._ensure_db_path()

        # Validate API key
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Create .env file with your API key."
            )

    def _find_config(self) -> Path:
        """
        Find config file with cascading search:
        1. ./gai/.gai.yaml (current directory subdirectory)
        2. ./.gai.yaml (current directory)
        3. ~/.config/gai/config.yaml (user config - XDG)
        4. /etc/gai/config.yaml (system-wide)

        Returns the first one found, or creates user config from defaults
        """
        search_paths = [
            Path.cwd() / "gai" / ".gai.yaml",                    # 1. ./gai/.gai.yaml
            Path.cwd() / ".gai.yaml",                            # 2. ./.gai.yaml
            self._get_user_config_dir() / "config.yaml",         # 3. ~/.config/gai/config.yaml
            Path("/etc/gai/config.yaml"),                        # 4. /etc/gai/config.yaml
        ]

        for path in search_paths:
            if path.exists():
                return path

        # No config found, use user config location (will be created on first save)
        return self._get_user_config_dir() / "config.yaml"

    def _get_user_config_dir(self) -> Path:
        """Get user config directory using XDG Base Directory spec"""
        xdg_config = os.getenv('XDG_CONFIG_HOME')
        if xdg_config:
            return Path(xdg_config) / 'gai'
        return Path.home() / '.config' / 'gai'

    def _load_env_cascade(self):
        """
        Load .env files with cascading priority:
        1. Global: ~/.config/gai/.env (loaded first)
        2. Local: ./.env (loaded second, overrides global)
        """
        # Load global env first
        global_env = self._get_user_config_dir() / '.env'
        if global_env.exists():
            load_dotenv(global_env)

        # Load local env (overrides global)
        local_env = Path.cwd() / '.env'
        if local_env.exists():
            load_dotenv(local_env, override=True)

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML config"""
        if not self.config_path.exists():
            return self._default_config()

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or self._default_config()

    def _ensure_db_path(self):
        """
        Set DB path relative to config file location if not specified.

        Logic:
        - If stats.db_path is null or not set → use config_dir/gai.db
        - If stats.db_path is relative → resolve relative to config_dir
        - If stats.db_path is absolute → use as-is
        """
        db_path = self.get('stats.db_path')

        if db_path is None:
            # Default: put DB in same directory as config file
            config_dir = self.config_path.parent
            default_db = config_dir / 'gai.db'
            self.set('stats.db_path', str(default_db))
        elif not Path(db_path).is_absolute():
            # Relative path: resolve relative to config directory
            config_dir = self.config_path.parent
            resolved_db = config_dir / db_path
            self.set('stats.db_path', str(resolved_db))
        # else: absolute path, use as-is

    def _default_config(self) -> Dict[str, Any]:
        """Default config if no file found"""
        return {
            "ai": {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "max_tokens": 30000,
                "fallback_models": [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant"
                ],
                "temperature": 0.3
            },
            "commit": {
                "format": "conventional",
                "auto_confirm": False,
                "scope_detection": True,
                "issue_detection": True,
                "multi_line": True,
                "max_diff_tokens": 30000
            },
            "review_merge": {
                "check_conflicts": True,
                "show_stats": True,
                "suggest_actions": True,
                "analyze_breaking_changes": True
            },
            "diff": {
                "summary_style": "detailed",
                "show_file_stats": True,
                "detect_patterns": True
            },
            "general": {
                "language": "en",
                "verbose": False,
                "stats_tracking": True
            },
            "stats": {
                "use_db": True,
                "backend": "sqlite",
                "db_path": None
            },
            "models": {
                "exclude_keywords": [
                    "whisper",
                    "tts",
                    "playai",
                    "distil-whisper"
                ]
            },
            "anannas": {
                "api_url": "https://api.anannas.ai/v1/chat/completions"
            },
            "ollama": {
                "base_url": "http://localhost:11434"
            }
        }

    def get(self, path: str, default: Any = None) -> Any:
        """Get config value by dot path (e.g., 'ai.model')"""
        keys = path.split('.')
        value = self.data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default

        return value if value is not None else default

    def set(self, path: str, value: Any):
        """Set config value by dot path"""
        keys = path.split('.')
        data = self.data

        for key in keys[:-1]:
            if key not in data:
                data[key] = {}
            data = data[key]

        data[keys[-1]] = value

    def save(self):
        """Save config to YAML file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config():
    """Reload config from files"""
    global _config
    _config = Config()
