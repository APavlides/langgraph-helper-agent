"""Configuration for the agent."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


class AgentMode(Enum):
    OFFLINE = "offline"
    ONLINE = "online"


def _get_env_or_yaml(env_key: str, yaml_val: Any, default: Any = None) -> Any:
    """Get value from env var, then yaml, then default."""
    if env_key and os.getenv(env_key):
        val = os.getenv(env_key)
        return type(default)(val) if isinstance(default, (int, float)) else val
    return yaml_val if yaml_val is not None else default


@dataclass
class Settings:
    """Agent configuration with priority: args > env > yaml > defaults."""

    config_path: str = "config.yaml"
    _config: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    mode: Optional[AgentMode] = None
    tavily_api_key: Optional[str] = None

    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    retrieval_k: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

    data_dir: Optional[Path] = None
    vectorstore_path: Optional[Path] = None

    rerank_threshold: Optional[float] = None
    max_web_results: int = 3

    def __post_init__(self):
        path = Path(self.config_path)
        self._config = yaml.safe_load(path.read_text()) if path.exists() else {}

        llm_cfg = self._config.get("llm", {})
        emb_cfg = self._config.get("embedding", {})
        agent_cfg = self._config.get("agent", {})
        data_cfg = self._config.get("data", {})

        self.llm_model = self.llm_model or _get_env_or_yaml(
            "LLM_MODEL", llm_cfg.get("model", {}).get("name"), "llama3.2:3b"
        )
        self.embedding_model = self.embedding_model or _get_env_or_yaml(
            "EMBEDDING_MODEL", emb_cfg.get("model"), "nomic-embed-text"
        )
        self.ollama_base_url = self.ollama_base_url or _get_env_or_yaml(
            "OLLAMA_BASE_URL",
            llm_cfg.get("ollama", {}).get("base_url") or emb_cfg.get("base_url"),
            "http://localhost:11434",
        )

        self.temperature = self.temperature or float(
            _get_env_or_yaml(
                "TEMPERATURE", llm_cfg.get("parameters", {}).get("temperature"), 0.1
            )
        )
        self.max_tokens = self.max_tokens or int(
            _get_env_or_yaml(
                "MAX_TOKENS", llm_cfg.get("parameters", {}).get("max_tokens"), 2000
            )
        )

        mode_str = _get_env_or_yaml("AGENT_MODE", agent_cfg.get("mode"), "offline")
        self.mode = self.mode or AgentMode(mode_str.lower())

        self.retrieval_k = self.retrieval_k or int(
            _get_env_or_yaml("RETRIEVAL_K", agent_cfg.get("retrieval_k"), 5)
        )
        self.chunk_size = self.chunk_size or int(
            _get_env_or_yaml("CHUNK_SIZE", agent_cfg.get("chunk_size"), 1000)
        )
        self.chunk_overlap = self.chunk_overlap or int(
            _get_env_or_yaml("CHUNK_OVERLAP", agent_cfg.get("chunk_overlap"), 200)
        )
        self.rerank_threshold = self.rerank_threshold or float(
            _get_env_or_yaml("RERANK_THRESHOLD", agent_cfg.get("rerank_threshold"), 0.0)
        )

        data_dir_str = _get_env_or_yaml("DATA_DIR", data_cfg.get("dir"), "data")
        self.data_dir = self.data_dir or Path(data_dir_str)

        vectorstore_str = _get_env_or_yaml(
            "VECTORSTORE_PATH", data_cfg.get("vectorstore"), "data/vectorstore"
        )
        self.vectorstore_path = self.vectorstore_path or Path(vectorstore_str)

        self.tavily_api_key = self.tavily_api_key or os.getenv("TAVILY_API_KEY")

        if self.mode == AgentMode.ONLINE and not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY required for online mode")

        self.data_dir.mkdir(parents=True, exist_ok=True)
