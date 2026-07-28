"""FastAPI wrapper around the agent."""
from fastapi import FastAPI
from pydantic import BaseModel
from agent import build_graph

app = FastAPI(title="Financial Complaint Signal & Search Agent")
_graph = build_graph()


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: QuestionRequest):
    result = _graph.invoke({
        "question": req.question,
        "retrieved_docs": [],
        "anomaly_data": {},
        "answer": "",
    })
    return {
        "answer": result["answer"],
        "retrieved_docs": result["retrieved_docs"],
        "anomaly_summary": result["anomaly_data"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
