"""Agent node functions."""

from langchain.chat_models.base import BaseChatModel
from sentence_transformers import CrossEncoder

from src.agent.state import AgentState

# Lazy load reranker to avoid startup cost
_reranker = None


def get_reranker():
    """Lazy load cross-encoder for reranking."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def create_retrieve_node(retriever):
    """Create retrieval node with reranking."""

    def retrieve(state: AgentState):
        query = state["messages"][-1].content if state["messages"] else ""
        # Get more candidates for reranking
        docs_with_scores = retriever.vectorstore.similarity_search_with_score(
            query, k=retriever.search_kwargs.get("k", 5) * 2  # Get 2x for reranking
        )

        # Rerank with cross-encoder
        reranker = get_reranker()
        doc_texts = [doc.page_content for doc, _ in docs_with_scores]
        rerank_scores = reranker.predict([(query, text) for text in doc_texts])

        # Sort by rerank scores (higher is better for cross-encoder)
        ranked_docs = sorted(
            zip(doc_texts, rerank_scores), key=lambda x: x[1], reverse=True
        )

        # Take top k after reranking
        k = retriever.search_kwargs.get("k", 5)
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


def route_after_retrieve(state: AgentState) -> str:
    """Route based on retrieval quality.

    Cross-encoder scores: higher = more relevant
    Typical: > 0.5 = good, 0.2-0.5 = questionable, < 0.2 = poor
    """
    if state["mode"] == "offline":
        return "generate"

    retrieval_score = state.get("retrieval_score", 0.0)

    if retrieval_score < 0.3:  # Low reranker confidence
        return "web_search_and_generate"

    return "generate"


def create_generate_node(llm: BaseChatModel):
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
        return {"messages": [response]}

    return generate


def create_web_search_and_generate_node(llm: BaseChatModel, search_tool):
    """Search web and generate with combined context."""

    def web_search_and_generate(state: AgentState):
        query = state["messages"][-1].content if state["messages"] else ""
        # Perform web search
        search_results = search_tool.invoke(query)
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

Question: {query}

Answer:"""
        response = llm.invoke(prompt)
        return {
            "messages": [response],
            "web_search_results": search_results,
        }

    return web_search_and_generate
