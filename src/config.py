import os
import re
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class AppConfig:
    active_provider: str
    fallback_provider: str
    providers: dict
    agent: dict
    memory: dict

    @classmethod
    def from_dict(cls, d: dict) -> "AppConfig":
        return cls(
            active_provider=d.get("active_provider", "zhipu"),
            fallback_provider=d.get("fallback_provider", "openai"),
            providers=d.get("providers", {}),
            agent={
                "max_iterations": 15,
                "context_compression_threshold": 0.7,
                "memory_nudge_interval": 10,
                **d.get("agent", {}),
            },
            memory={
                "data_dir": "~/.easy_agent",
                "memory_md_max_chars": 2000,
                "user_md_max_chars": 1500,
                "fts5_retention_days": 90,
                "vector_max_entries": 500,
                **d.get("memory", {}),
            },
        )


def _substitute_env(value: str) -> str:
    """Replace ${ENV_VAR} patterns with environment variable values."""
    pattern = re.compile(r"\$\{(\w+)\}")
    return pattern.sub(lambda m: os.environ.get(m.group(1), ""), value)


def _substitute_dict(d: dict) -> dict:
    """Recursively substitute env vars in dict values."""
    result = {}
    for k, v in d.items():
        if isinstance(v, str):
            result[k] = _substitute_env(v)
        elif isinstance(v, dict):
            result[k] = _substitute_dict(v)
        else:
            result[k] = v
    return result


def load_config(path: str = None) -> AppConfig:
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = _substitute_dict(raw)
    return AppConfig.from_dict(raw)