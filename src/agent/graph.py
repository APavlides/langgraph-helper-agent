"""Agent graph construction."""

from typing import Any, cast

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    create_generate_node,
    create_retrieve_node,
    create_route_after_retrieve,
    create_web_search_and_generate_node,
)
from src.agent.state import AgentState
from src.config import AgentMode, Settings


def create_retriever(settings: Settings) -> Any:
    if settings.vectorstore_path is None:
        raise ValueError("VECTORSTORE_PATH is not configured")
    if settings.embedding_model is None:
        raise ValueError("EMBEDDING_MODEL is not configured")
    if settings.ollama_base_url is None:
        raise ValueError("OLLAMA_BASE_URL is not configured")

    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

    if not settings.vectorstore_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {settings.vectorstore_path}. "
            "Run 'python scripts/build_vectorstore.py' first."
        )

    vectorstore = FAISS.load_local(
        str(settings.vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_k or 5},
    )


def create_llm(settings: Settings) -> ChatOllama:
    if settings.llm_model is None:
        raise ValueError("LLM_MODEL is not configured")
    if settings.ollama_base_url is None:
        raise ValueError("OLLAMA_BASE_URL is not configured")

    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )


def create_search_tool(settings: Settings) -> Any:
    if settings.mode != AgentMode.ONLINE:
        return None

    if settings.tavily_api_key:
        from langchain_tavily import TavilySearch

        return TavilySearch(
            api_key=settings.tavily_api_key,
            max_results=settings.max_web_results,
        )

    return None


def create_agent(settings: Settings) -> Any:
    retriever = create_retriever(settings)
    llm = create_llm(settings)
    search_tool = create_search_tool(settings)

    retrieve_node = create_retrieve_node(retriever)
    generate_node = create_generate_node(llm)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", cast(Any, retrieve_node))
    graph.add_node("generate", cast(Any, generate_node))

    if settings.mode == AgentMode.ONLINE and search_tool:
        web_search_and_generate_node = create_web_search_and_generate_node(
            llm, search_tool
        )
        graph.add_node(
            "web_search_and_generate", cast(Any, web_search_and_generate_node)
        )

    graph.add_edge(START, "retrieve")

    if settings.mode == AgentMode.ONLINE and search_tool:
        # Route after retrieve based on retrieval quality
        graph.add_conditional_edges(
            "retrieve",
            create_route_after_retrieve(settings.rerank_threshold or 0.0),
            {
                "generate": "generate",
                "web_search_and_generate": "web_search_and_generate",
            },
        )
        graph.add_edge("web_search_and_generate", END)
    else:
        graph.add_edge("retrieve", "generate")

    graph.add_edge("generate", END)

    return graph.compile()


def visualize_graph(settings: Settings) -> str:
    return str(create_agent(settings).get_graph().draw_mermaid())
