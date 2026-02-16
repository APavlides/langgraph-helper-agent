import os
import uuid
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langsmith import Client, evaluate

# from langsmith.evaluation import LangChainStringEvaluator, evaluate  <-- REMOVED
from langsmith.schemas import Example, Run

from src.agent.graph import create_agent
from src.config import AgentMode, Settings

# --- 1. CONFIGURATION ---
DATASET_NAME = "LangGraph Agent Trajectory Eval"
# Use local Ollama as the judge since external APIs are flaky
JUDGE_LLM = ChatOllama(
    model="llama3.2:3b", base_url="http://host.docker.internal:11434", temperature=0
)


# --- 2. DATASET CREATION ---
def create_dataset():
    client = Client()

    # Check if exists
    try:
        if client.has_dataset(dataset_name=DATASET_NAME):
            # For this demo, we'll reuse it or delete it. Let's reuse.
            print(f"Dataset '{DATASET_NAME}' already exists.")
            return client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        pass

    ds = client.create_dataset(
        dataset_name=DATASET_NAME, description="Evaluating Agentic RAG patterns"
    )

    examples = [
        # Type A: Direct Query (Should pass through Retrieve -> Grade -> Generate)
        {
            "inputs": {"question": "How do I define a StateGraph?"},
            "outputs": {
                "answer": "You define a StateGraph by initializing the StateGraph class with a state schema (TypedDict), adding nodes with graph.add_node, and adding edges.",
                "required_path": ["retrieve", "generate"],
            },
        },
        # Type B: Ambiguous Query (Should trigger Rewrite -> Retrieve -> Grade -> Generate)
        {
            "inputs": {"question": "memory"},
            "outputs": {
                "answer": "In LangGraph, memory refers to persistence, which is handled by checkpointers (like MemorySaver or PostgresSaver). It allows resuming state across threads.",
                "required_path": ["rewrite", "retrieve", "generate"],
            },
        },
    ]

    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=ds.id,
    )
    return ds


# --- 3. THE SYSTEM UNDER TEST (TARGET) ---
def agent_target(inputs: dict) -> dict:
    """The function that runs our agent for evaluation."""
    # Ensure env vars are set for local execution
    # In docker, these are set. If running locally, you might need to export them.
    # We assume this script runs inside 'docker compose run agent-offline' or similar environment.

    try:
        # FORCE OLLAMA BASE URL for EVAL - ensuring it points to host from inside container
        forced_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
        )

        settings = Settings(mode=AgentMode.OFFLINE, ollama_base_url=forced_url)
        agent = create_agent(settings)

        # Invoke the graph using stream to capture the trajectory
        inputs = {"messages": [HumanMessage(content=inputs["question"])]}
        trajectory = []
        final_state = {}

        # We need the final state for the answer/context, but want to track nodes visited
        # Using stream_mode="updates" gives us the node name and its output
        for output in agent.stream(inputs, stream_mode="updates"):
            for node_name, state_update in output.items():
                trajectory.append(node_name)
                # We can try to maintain state, but for the purpose of this eval,
                # we primarily need the final answer from the generation node
                if "messages" in state_update:
                    # This is a simplification; in a real app better state merging is needed
                    final_state.update(state_update)
                if "retrieved_contexts" in state_update:
                    final_state["retrieved_contexts"] = state_update[
                        "retrieved_contexts"
                    ]

        # If we didn't get a final state (e.g. error), fail gently
        if not final_state:
            return {"answer": "Error: No output generated", "trajectory": []}

        # Extract answer from the last message in the update
        # Depending on the graph, the final "generate" node updates 'messages'
        # with the AIMessage.
        messages = final_state.get("messages", [])
        last_message = messages[-1] if messages else None
        answer_text = last_message.content if last_message else "No answer"

        # Format output for the evaluators
        return {
            "answer": answer_text,
            "context": final_state.get("retrieved_contexts", []),
            "trajectory": trajectory,  # Explicitly return the path
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "error": str(e), "trajectory": []}


# --- 4. EVALUATORS ---


# Metrics 1: Result Correctness (LLM-as-a-Judge)
# Functional evaluator instead of broken class
def correctness_evaluator(run: Run, example: Example) -> dict:
    # Use direct LLM call instead of load_evaluator which is missing
    target = example.outputs["answer"]
    pred = run.outputs["answer"]
    input_q = example.inputs["question"]

    prompt = f"""You are an evaluator.
    Question: {input_q}
    Ref Answer: {target}
    Prediction: {pred}
    
    Is the prediction correct based on the reference? Return 'score: 1' or 'score: 0' and explanation."""

    res = JUDGE_LLM.invoke(prompt)
    content = res.content.lower()
    score = 1 if "score: 1" in content else 0

    return {
        "key": "correctness",
        "score": score,
        "comment": str(res.content),
    }


# Metrics 2: Single Step / Trajectory Check (White-box via 'trajectory' output)
def trajectory_evaluator(run: Run, example: Example) -> dict:
    """
    Checks if the actual execution path (nodes visited) matched the expectation.
    """
    expected_path = example.outputs.get("required_path", [])

    # Retrieve the explicitly captured trajectory from the run output
    # This avoids the complexity of parsing the trace tree asynchronously
    actual_path = run.outputs.get("trajectory", [])

    # Check if expected path acts as a subsequence or subset
    # For this strict test, let's check if expected nodes appear in order

    missing_steps = []

    # Simple check: filter actual path to only include interesting nodes
    # This handles cases where we might have extra steps like 'start' or 'end'
    # though our stream only yields node names like 'retrieve', 'generate'.

    # Verify order
    current_idx = 0
    found_all = True

    for needed_node in expected_path:
        if needed_node in actual_path[current_idx:]:
            # Find the first occurrence after current_idx
            current_idx += actual_path[current_idx:].index(needed_node) + 1
        else:
            missing_steps.append(needed_node)
            found_all = False

    score = 1 if found_all else 0
    return {
        "key": "trajectory_faithfulness",
        "score": score,
        "comment": f"Path: {actual_path} | Missing: {missing_steps}",
    }


# --- 5. RUN EVALUATION ---
if __name__ == "__main__":
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("❌ EVALUATION SKIPPED: LANGSMITH_API_KEY not found in environment.")
        print("Please add it to your .env file.")
        exit(1)

    print("🚀 Starting LangSmith Evaluation...")
    dataset = create_dataset()

    results = evaluate(
        agent_target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, trajectory_evaluator],
        experiment_prefix="agentic-rag-test",
    )

    print("\n✅ Evaluation complete! View results at the URL above.")
