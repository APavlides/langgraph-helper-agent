"""Agent graph construction."""

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (create_generate_node, create_regenerate_node,
                             create_retrieve_node, create_web_search_node,
                             route_after_generate)
from src.agent.state import AgentState
from src.config import AgentMode, Settings


def create_retriever(settings: Settings):
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
        search_kwargs={"k": settings.retrieval_k},
    )


def create_llm(settings: Settings):
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )


def create_search_tool(settings: Settings):
    if settings.mode != AgentMode.ONLINE:
        return None
    
    if settings.tavily_api_key:
        from langchain_tavily import TavilySearch
        return TavilySearch(
            api_key=settings.tavily_api_key,
            max_results=settings.max_web_results,
        )
    
    from langchain_community.tools import DuckDuckGoSearchResults
    return DuckDuckGoSearchResults(max_results=settings.max_web_results)


def create_agent(settings: Settings):
    retriever = create_retriever(settings)
    llm = create_llm(settings)
    search_tool = create_search_tool(settings)
    
    retrieve_node = create_retrieve_node(retriever)
    generate_node = create_generate_node(llm, settings.confidence_threshold)
    
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    if settings.mode == AgentMode.ONLINE and search_tool:
        web_search_node = create_web_search_node(search_tool)
        regenerate_node = create_regenerate_node(llm)
        graph.add_node("web_search", web_search_node)
        graph.add_node("regenerate", regenerate_node)
    
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    
    if settings.mode == AgentMode.ONLINE and search_tool:
        graph.add_conditional_edges(
            "generate",
            route_after_generate,
            {"web_search": "web_search", "__end__": END}
        )
        graph.add_edge("web_search", "regenerate")
        graph.add_edge("regenerate", END)
    else:
        graph.add_edge("generate", END)
    
    return graph.compile()


def visualize_graph(settings: Settings) -> str:
    return create_agent(settings).get_graph().draw_mermaid()
