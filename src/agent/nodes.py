"""Agent node functions."""

from langchain.chat_models.base import BaseChatModel

from src.agent.state import AgentState


def create_retrieve_node(retriever):
    """Create retrieval node."""
    def retrieve(state: AgentState):
        query = state["messages"][-1].content if state["messages"] else ""
        docs = retriever.invoke(query)
        return {"retrieved_contexts": [doc.page_content for doc in docs]}
    return retrieve


def create_generate_node(llm: BaseChatModel, confidence_threshold: float):
    """Create generation node."""
    def generate(state: AgentState):
        query = state["messages"][-1].content if state["messages"] else ""
        context = "\n\n".join(state["retrieved_contexts"])
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        response = llm.invoke(prompt)
        return {
            "messages": [response],
            "confidence_score": 0.9,
            "needs_web_search": False,
        }
    return generate


def create_web_search_node(search_tool):
    """Create web search node."""
    def web_search(state: AgentState):
        query = state["messages"][-1].content if state["messages"] else ""
        results = search_tool.invoke(query)
        return {"web_search_results": results}
    return web_search


def create_regenerate_node(llm: BaseChatModel):
    """Create regeneration node with web results."""
    def regenerate(state: AgentState):
        query = state["messages"][-1].content if state["messages"] else ""
        web_results = "\n\n".join(state.get("web_search_results", []) or [])
        context = "\n\n".join(state["retrieved_contexts"])
        prompt = f"""Answer the question using both the context and web results.

Context:
{context}

Web Results:
{web_results}

Question: {query}

Answer:"""
        response = llm.invoke(prompt)
        return {"messages": [response]}
    return regenerate


def route_after_generate(state: AgentState) -> str:
    """Route based on mode and confidence."""
    if state["mode"] == "offline":
        return "__end__"
    if state.get("needs_web_search"):
        return "web_search"
    return "__end__"

