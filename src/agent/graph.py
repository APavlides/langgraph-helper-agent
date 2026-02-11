"""Agent graph construction."""

from typing import Any, cast

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    create_generate_node,
    create_grade_documents_node,
    create_retrieve_node,
    create_rewrite_query_node,
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
    # 1. Load Vector (FAISS)
    retriever_with_vs = create_retriever(settings)
    # create_retriever now needs to return the underlying vectorstore so we can get docs for BM25
    # But create_retriever returns vectorstore.as_retriever(), which hides the store.
    # Let's refactor that function slightly or just access it if possible.
    # Wait, retriever.vectorstore is accessible on the object returned by as_retriever()

    vector_retriever = retriever_with_vs

    # 2. Load Keyword (BM25)
    # We extract documents from the vectorstore to build the keyword index
    # validation: accessing private attribute _dict is risky but standard for FAISS in LangChain
    try:
        if hasattr(vector_retriever, "vectorstore") and hasattr(
            vector_retriever.vectorstore, "docstore"
        ):
            docs = list(vector_retriever.vectorstore.docstore._dict.values())
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = settings.retrieval_k or 5
        else:
            bm25_retriever = None
    except Exception:
        # Fallback if docstore access fails
        bm25_retriever = None

    llm = create_llm(settings)
    search_tool = create_search_tool(settings)

    rewrite_query_node = create_rewrite_query_node(llm)
    retrieve_node = create_retrieve_node(vector_retriever, bm25_retriever)
    grade_documents_node = create_grade_documents_node(llm)
    generate_node = create_generate_node(llm)

    graph = StateGraph(AgentState)
    graph.add_node("rewrite", cast(Any, rewrite_query_node))
    graph.add_node("retrieve", cast(Any, retrieve_node))
    graph.add_node("grade_documents", cast(Any, grade_documents_node))
    graph.add_node("generate", cast(Any, generate_node))

    if settings.mode == AgentMode.ONLINE and search_tool:
        web_search_and_generate_node = create_web_search_and_generate_node(
            llm, search_tool
        )
        graph.add_node(
            "web_search_and_generate", cast(Any, web_search_and_generate_node)
        )

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    def check_relevance(state: AgentState) -> str:
        """
        Check if we found any relevant documents.
        If yes -> Generate
        If no -> Rewrite Query (Loop) or Web Search (if online)
        """
        if state["retrieved_contexts"]:
            return "generate"

        if settings.mode == AgentMode.ONLINE and search_tool:
            print("---DECISION: DOCS IRRELEVANT, FALLING BACK TO WEB SEARCH---")
            return "web_search_and_generate"

        print("---DECISION: ALL DOCS IRRELEVANT, LOOPING BACK---")
        return "rewrite"

    check_relevance_map = {
        "generate": "generate",
        "rewrite": "rewrite",
    }

    if settings.mode == AgentMode.ONLINE and search_tool:
        check_relevance_map["web_search_and_generate"] = "web_search_and_generate"

    graph.add_conditional_edges(
        "grade_documents",
        check_relevance,
        check_relevance_map,
    )

    if settings.mode == AgentMode.ONLINE and search_tool:
        graph.add_edge("web_search_and_generate", END)

    graph.add_edge("generate", END)

    return graph.compile()


def visualize_graph(settings: Settings) -> str:
    return str(create_agent(settings).get_graph().draw_mermaid())
