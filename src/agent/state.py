"""State schema for the LangGraph Helper Agent."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State schema for the agent graph.

    This TypedDict defines the structure of data that flows through
    the LangGraph agent. Each node can read from and write to this state.

    Attributes:
        messages: Conversation history with automatic message aggregation.
            Uses LangGraph's add_messages reducer to automatically append
            new messages rather than replacing the list.

        retrieved_contexts: List of document chunks retrieved from the
            vector store. These provide the context for answering questions.

        retrieval_score: Average cross-encoder reranking score.
            Higher scores indicate better matches; negative scores are poor.

        mode: Current operating mode - either "offline" (uses only local
            vector store) or "online" (can use web search for additional info).

        web_search_results: Results from web search, if performed.
            Only populated in online mode when retrieval quality is low.
    """

    # Core conversation state
    messages: Annotated[list[BaseMessage], add_messages]

    # Retrieval state
    retrieved_contexts: list[str]
    retrieval_score: float | None

    # Mode and routing state
    mode: Literal["offline", "online"]

    # Web search state (online mode only)
    web_search_results: list[str] | None


def create_initial_state(mode: Literal["offline", "online"]) -> AgentState:
    """
    Create an initial state for a new conversation.

    Args:
        mode: Operating mode for this conversation

    Returns:
        Initial AgentState with empty/default values
    """
    return AgentState(
        messages=[],
        retrieved_contexts=[],
        retrieval_score=None,
        mode=mode,
        web_search_results=None,
    )
