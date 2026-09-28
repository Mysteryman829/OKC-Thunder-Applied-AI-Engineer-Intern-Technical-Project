"""FastAPI chat endpoint over the shared retrieval and reasoning in backend/rag.py.

usage:
    uvicorn backend.server:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sqlalchemy as sa

from backend.config import DB_DSN
from backend.rag import answer_chat

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
eng = sa.create_engine(DB_DSN)


class Q(BaseModel):
    question: str


@app.post("/api/chat")
def answer(q: Q) -> dict:
    """Answer a free-text question using the batch pipeline's reasoning and citations.

    :param q: request body with a single "question" field
    :return: {"answer": str, "evidence": [{"table", "id", "doc"}]}
    """
    print("Received question")
    with eng.begin() as cx:
        return answer_chat(cx, q.question)
