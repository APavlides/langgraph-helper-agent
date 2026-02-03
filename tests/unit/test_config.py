"""Unit tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from src.config import AgentMode, Settings


class TestAgentMode:
    """Tests for AgentMode enum."""
    
    def test_offline_mode(self):
        assert AgentMode.OFFLINE.value == "offline"
    
    def test_online_mode(self):
        assert AgentMode.ONLINE.value == "online"
    
    def test_mode_from_string(self):
        assert AgentMode("offline") == AgentMode.OFFLINE
        assert AgentMode("online") == AgentMode.ONLINE


class TestSettings:
    """Tests for Settings configuration."""
    
    @patch.dict(os.environ, {
        "AGENT_MODE": "offline",
        "LLM_MODEL": "llama3.2:3b",
        "EMBEDDING_MODEL": "nomic-embed-text",
        "OLLAMA_BASE_URL": "http://localhost:11434",
    })
    def test_settings_from_env(self):
        settings = Settings()
        assert settings.mode == AgentMode.OFFLINE
        assert settings.llm_model == "llama3.2:3b"
        assert settings.embedding_model == "nomic-embed-text"
        assert settings.ollama_base_url == "http://localhost:11434"
    
    @patch.dict(os.environ, {
        "AGENT_MODE": "online",
        "TAVILY_API_KEY": "tavily-key",
    })
    def test_online_mode_settings(self):
        settings = Settings()
        assert settings.mode == AgentMode.ONLINE
        assert settings.tavily_api_key == "tavily-key"
    
    @patch.dict(os.environ, {
        "AGENT_MODE": "online",
    }, clear=True)
    def test_online_mode_without_tavily_raises(self):
        with pytest.raises(ValueError, match="TAVILY_API_KEY required"):
            Settings()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_default_values(self):
        settings = Settings()
        assert settings.llm_model == "llama3.2:3b"
        assert settings.embedding_model == "nomic-embed-text"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.temperature == 0.1
        assert settings.retrieval_k == 5
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
