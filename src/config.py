import os
import re
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class KnowledgeIndexConfig:
    path: str
    description: str


@dataclass
class KnowledgeBaseConfig:
    base_dir: str
    indexes: dict[str, KnowledgeIndexConfig]


@dataclass
class MCPServerConfig:
    url: str
    timeout: int = 30
    intent: str = ""                        # 该 MCP 对应的意图类型
    endpoints: list[dict] = field(default_factory=list)  # 端点描述


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class BeautyConfig:
    knowledge_base: KnowledgeBaseConfig
    mcp_servers: dict[str, MCPServerConfig]
    intent_prompt: str
    web: WebConfig

    @classmethod
    def from_dict(cls, d: dict) -> "BeautyConfig":
        kb_data = d.get("knowledge_base", {})
        indexes = {
            name: KnowledgeIndexConfig(**idx)
            for name, idx in kb_data.get("indexes", {}).items()
        }
        knowledge_base = KnowledgeBaseConfig(
            base_dir=kb_data.get("base_dir", ""),
            indexes=indexes,
        )

        mcp_servers = {
            name: MCPServerConfig(**server)
            for name, server in d.get("mcp_servers", {}).items()
        }

        web_data = d.get("web", {})
        web = WebConfig(
            host=web_data.get("host", "0.0.0.0"),
            port=web_data.get("port", 8080),
        )

        return cls(
            knowledge_base=knowledge_base,
            mcp_servers=mcp_servers,
            intent_prompt=d.get("intent_prompt", ""),
            web=web,
        )


@dataclass
class AppConfig:
    active_provider: str
    fallback_provider: str
    providers: dict
    agent: dict
    memory: dict
    search: dict
    beauty: BeautyConfig | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "AppConfig":
        beauty = None
        if "beauty" in d:
            beauty = BeautyConfig.from_dict(d["beauty"])

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
            search=d.get("search", {}),
            beauty=beauty,
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Config file is empty or invalid: {path}")
    raw = _substitute_dict(raw)
    cfg = AppConfig.from_dict(raw)

    # Set search-related env vars from config
    search_cfg = cfg.search
    if search_cfg.get("tavily_api_key"):
        os.environ["TAVILY_API_KEY"] = search_cfg["tavily_api_key"]

    return cfg