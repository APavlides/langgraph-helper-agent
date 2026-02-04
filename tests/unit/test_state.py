"""Tests for agent state management."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.state import AgentState


def test_agent_state_initialization():
    """Test agent state can be initialized."""
    messages = [HumanMessage(content="How do I use persistence?")]
    state = AgentState(
        messages=messages,
        retrieved_contexts=["context about persistence"],
        mode="offline",
        needs_web_search=False,
        confidence_score=0.85,
        web_search_results=None,
    )
    assert len(state["messages"]) == 1
    assert state["mode"] == "offline"
    assert state["confidence_score"] == 0.85


def test_agent_state_with_multiple_contexts():
    """Test state with multiple retrieved contexts."""
    state = AgentState(
        messages=[HumanMessage(content="test question")],
        retrieved_contexts=[
            "context 1 about the topic",
            "context 2 with more details",
            "context 3 with examples",
        ],
        mode="offline",
        needs_web_search=False,
        confidence_score=0.9,
        web_search_results=None,
    )
    assert len(state["retrieved_contexts"]) == 3
    assert state["confidence_score"] == 0.9


def test_agent_state_online_mode():
    """Test state in online mode."""
    state = AgentState(
        messages=[HumanMessage(content="What's new?")],
        retrieved_contexts=["local context"],
        mode="online",
        needs_web_search=True,
        confidence_score=0.5,
        web_search_results=["web result 1", "web result 2"],
    )
    assert state["mode"] == "online"
    assert state["needs_web_search"] is True
    assert len(state["web_search_results"]) == 2
