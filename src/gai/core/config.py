"""
Configuration management for gai
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Manages gai configuration from .env and .gai.yaml"""

    def __init__(self, config_path: Optional[Path] = None):
        # Load .env first
        load_dotenv()

        # Find config file
        if config_path is None:
            config_path = self._find_config()

        self.config_path = config_path
        self.data = self._load_yaml()

        # Validate API key
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Create .env file with your API key."
            )

    def _find_config(self) -> Path:
        """Find .gai.yaml in current dir or parent dirs"""
        current = Path.cwd()

        # Check current dir
        config = current / ".gai.yaml"
        if config.exists():
            return config

        # Check parent dirs up to home
        for parent in current.parents:
            config = parent / ".gai.yaml"
            if config.exists():
                return config

            # Stop at home dir
            if parent == Path.home():
                break

        # Use package default
        package_dir = Path(__file__).parent.parent
        return package_dir / ".gai.yaml"

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML config"""
        if not self.config_path.exists():
            return self._default_config()

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or self._default_config()

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
