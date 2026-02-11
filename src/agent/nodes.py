"""Agent node functions."""

from collections.abc import Callable
from typing import Any, cast

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

from src.agent.state import AgentState

# Lazy load reranker to avoid startup cost
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Lazy load cross-encoder for reranking."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert _reranker is not None
    return _reranker


def create_rewrite_query_node(
    llm: BaseChatModel,
) -> Callable[[AgentState], dict[str, Any]]:
    """Rewrite the user question to be optimized for vector retrieval."""

    def rewrite_query(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        original_query = str(last_message.content)

        system_prompt = """You are a query rewriter for an AI assistant focused on LangGraph and LangChain libraries.
Your goal is to convert user questions into search queries optimized for a technical documentation vector store.

Rules:
1. Always interpret ambiguous terms like "memory", "graph", "state", "nodes" in the context of LangGraph/LangChain Python libraries.
2. "Memory" usually means "Persistence" or "Checkpointers".
3. Remove filler words (please, help me, etc.).
4. Expand acronyms if clear (e.g., "LC" -> "LangChain").
5. Output ONLY the rewritten query, no explanations."""

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Original Question: {original_query}"),
            ]
        )

        rewritten = str(response.content)
        print(f"\n🔄 Rewritten Query: '{rewritten}'")
        return {"search_query": rewritten}

    return rewrite_query


def create_retrieve_node(
    retriever: Any, bm25_retriever: Any | None = None
) -> Callable[[AgentState], dict[str, Any]]:
    """Create retrieval node with Hybrid Search (Vector + Keyword) and Reranking."""

    def retrieve(state: AgentState) -> dict[str, Any]:
        # Use the rewritten query if available, otherwise fallback to raw input
        query_source = state.get("search_query")
        if not query_source and state["messages"]:
            query_source = state["messages"][-1].content

        query = str(query_source) if query_source else ""

        # 1. VECTOR SEARCH (Semantic)
        # Get 2x candidates for reranking
        k = retriever.search_kwargs.get("k", 5)

        # Note: We ignore the vector scores here because we are about to rerank everything
        vector_docs_with_scores = cast(
            list[tuple[Any, float]],
            retriever.vectorstore.similarity_search_with_score(query, k=k * 2),
        )
        vector_docs = [doc for doc, _ in vector_docs_with_scores]

        # 2. KEYWORD SEARCH (Exact Match)
        keyword_docs = []
        if bm25_retriever:
            # BM25 usually creates a sparse index good for exact variable names
            keyword_docs = bm25_retriever.invoke(query)
            # Limit precision if needed, though reranker handles it
            keyword_docs = keyword_docs[: k * 2]

        # 3. HYBRID FUSION (Union + Deduplication)
        # Combine duplicates based on page_content
        combined_docs_map = {
            doc.page_content: doc for doc in vector_docs + keyword_docs
        }
        unique_docs = list(combined_docs_map.values())

        # 4. RERANKING (Cross-Encoder)
        reranker = get_reranker()
        doc_texts: list[str] = [str(doc.page_content) for doc in unique_docs]

        if not doc_texts:
            return {"retrieved_contexts": [], "retrieval_score": 0.0}

        pair_inputs = [(query, text) for text in doc_texts]
        rerank_scores = cast(Any, reranker).predict(pair_inputs)

        # Sort by rerank scores (higher is better for cross-encoder)
        ranked_docs = sorted(
            zip(doc_texts, rerank_scores, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )

        # Take top k after reranking
        top_docs = ranked_docs[:k]
        contexts = [doc for doc, _ in top_docs]
        avg_score = (
            sum(score for _, score in top_docs) / len(top_docs) if top_docs else 0.0
        )

        return {
            "retrieved_contexts": contexts,
            "retrieval_score": avg_score,
        }

    return retrieve


def create_grade_documents_node(
    llm: BaseChatModel,
) -> Callable[[AgentState], dict[str, Any]]:
    """Filter retrieved documents for relevance to the question."""

    def grade_documents(state: AgentState) -> dict[str, Any]:
        # We use the rewritten query for grading since that's what we searched for
        query = state.get("search_query", state["messages"][-1].content)
        documents = state["retrieved_contexts"]

        if not documents:
            return {"retrieved_contexts": []}

        # Validate with LLM
        filtered_docs = []

        system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        for doc in documents:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Retrieved document: \n\n {doc} \n\n User Question: {query} \n\n Is this relevant? dictionary with 'score' key containing 'yes' or 'no'"
                ),
            ]
            response = llm.invoke(messages)
            grade = response.content.lower()

            # Simple check for 'yes' in response - LLMs can be chatty
            if "yes" in grade:
                filtered_docs.append(doc)
            else:
                print("creating grade_doc_node: Document filtered out as irrelevant")

        return {"retrieved_contexts": filtered_docs}

    return grade_documents


def create_route_after_retrieve(rerank_threshold: float) -> Callable[[AgentState], str]:
    """Create routing function based on retrieval quality.

    Cross-encoder scores: higher = more relevant
    Typical: > 0.5 = good, 0.2-0.5 = questionable, < 0.0 = poor
    """

    def route_after_retrieve(state: AgentState) -> str:
        if state["mode"] == "offline":
            return "generate"

        retrieval_score = state.get("retrieval_score")
        if retrieval_score is None:
            retrieval_score = 0.0

        if retrieval_score < rerank_threshold:
            return "web_search_and_generate"

        return "generate"

    return route_after_retrieve


def create_generate_node(llm: BaseChatModel) -> Callable[[AgentState], dict[str, Any]]:
    """Create generation node."""

    def generate(state: AgentState) -> dict[str, Any]:
        query = state["messages"][-1].content if state["messages"] else ""
        context = "\n\n".join(state["retrieved_contexts"])
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        response = llm.invoke(prompt)
        return {"messages": [response]}

    return generate


def create_web_search_and_generate_node(
    llm: BaseChatModel, search_tool: Any
) -> Callable[[AgentState], dict[str, Any]]:
    """Search web and generate with combined context."""

    def web_search_and_generate(state: AgentState) -> dict[str, Any]:
        # Use rewritten query for search if available, but keep original for response context
        search_query = state.get("search_query")
        if not search_query and state["messages"]:
            search_query = state["messages"][-1].content

        original_query = state["messages"][-1].content if state["messages"] else ""

        # Perform web search
        search_results = search_tool.invoke(str(search_query) if search_query else "")
        web_results = (
            "\n\n".join(search_results)
            if isinstance(search_results, list)
            else str(search_results)
        )
        context = "\n\n".join(state["retrieved_contexts"])
        prompt = f"""Answer the question using both the context and web results.

Context:
{context}

Web Results:
{web_results}

Question: {original_query}

Answer:"""
        response = llm.invoke(prompt)
        return {
            "messages": [response],
            "web_search_results": search_results,
        }

    return web_search_and_generate
