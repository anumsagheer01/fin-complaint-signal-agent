"""
LangGraph agent that orchestrates two tools:
  1. retrieve_complaints - TF-IDF search over the complaint narrative corpus
  2. check_anomalies - looks up anomaly flags for a category/date range

The agent routes a user question to the right tool(s), then generates a
cited answer using the Anthropic API.

NOTE: generation requires an Anthropic API key. Set it as an environment
variable before running:
    export ANTHROPIC_API_KEY=your_key_here
"""
import os
import json
from pathlib import Path
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from anthropic import Anthropic

from retrieval import load_index, retrieve as retrieve_docs

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

_index = None
_anomaly_summary = None


def get_index():
    global _index
    if _index is None:
        _index = load_index()
    return _index


def get_anomaly_summary():
    global _anomaly_summary
    if _anomaly_summary is None:
        with open(DATA_DIR / "anomaly_summary.json") as f:
            _anomaly_summary = json.load(f)
    return _anomaly_summary


# ---- Tools ----

def tool_retrieve_complaints(query: str, k: int = 5):
    index = get_index()
    return retrieve_docs(query, index, k=k)


def tool_check_anomalies(category: str = None):
    summary = get_anomaly_summary()
    if category and category in summary:
        return {category: summary[category]}
    return summary


# ---- Agent state ----

class AgentState(TypedDict):
    question: str
    retrieved_docs: list
    anomaly_data: dict
    answer: str


def route_and_retrieve(state: AgentState) -> AgentState:
    docs = tool_retrieve_complaints(state["question"], k=5)
    anomalies = tool_check_anomalies()
    return {**state, "retrieved_docs": docs, "anomaly_data": anomalies}


def generate_answer(state: AgentState) -> AgentState:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        state["answer"] = (
            "[No ANTHROPIC_API_KEY set — skipping generation. "
            "Retrieved docs and anomaly data are available in state for inspection.]"
        )
        return state

    client = Anthropic(api_key=api_key)

    docs_context = "\n\n".join(
        f"[{d['doc_id']}] Category: {d['category']} | Issue: {d['issue']}\n{d['narrative_snippet']}"
        for d in state["retrieved_docs"]
    )
    anomaly_context = json.dumps(state["anomaly_data"], indent=2)

    prompt = f"""You are a servicing insights assistant for a financial institution.
Answer the question using ONLY the retrieved complaint documents and anomaly data below.
Cite document IDs (e.g. [doc_000123]) when referencing specific complaints.

Question: {state['question']}

Retrieved complaint documents:
{docs_context}

Anomaly detection summary:
{anomaly_context}

Answer:"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    state["answer"] = response.content[0].text
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", route_and_retrieve)
    graph.add_node("generate", generate_answer)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "question": "Why are customers complaining about account closures?",
        "retrieved_docs": [],
        "anomaly_data": {},
        "answer": "",
    })
    print("=== Retrieved docs ===")
    for d in result["retrieved_docs"]:
        print(f"  {d['doc_id']} ({d['category']} / {d['issue']}) score={d['score']:.3f}")
    print("\n=== Anomaly data ===")
    print(json.dumps(result["anomaly_data"], indent=2)[:500])
    print("\n=== Answer ===")
    print(result["answer"])
