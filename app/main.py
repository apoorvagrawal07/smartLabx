from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.middleware.request_context import (
    request_context_middleware
)

from app.graph.workflow import build_graph
from app.rag.rag_service import RAGService
from app.rag.retriever import Retriever
from app.services.model_gateway import ModelGateway
from app.tools.tool_executor import ToolExecutor


app = FastAPI()

app.middleware("http")(
    request_context_middleware
)

model_gateway = ModelGateway()


class LazyRetriever:

    def __init__(self):
        self._retriever = None

    def retrieve(self, query, limit=3):
        if self._retriever is None:
            self._retriever = Retriever()

        return self._retriever.retrieve(query, limit)


workflow = build_graph(
    model_gateway=model_gateway,
    rag_service=RAGService(
        retriever=LazyRetriever(),
        model_gateway=model_gateway,
    ),
    tool_executor=ToolExecutor(),
)


class ChatRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000
    )
    conversation_id: str

    mode: str = "fast"

class ChatResponse(BaseModel):

    answer: str
    conversation_id: str


@app.post("/chat")
def chat(request: ChatRequest):
    """Continue a conversation by ID; history is lost when the server restarts."""

    config = {
        "configurable": {
            "thread_id": request.conversation_id
        }
    }

    checkpoint = workflow.get_state(config)
    history = checkpoint.values.get("messages", [])

    result = workflow.invoke(
        {
            "messages": history + [
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            "user_message": request.message,
            "iteration_count": 0,
        },
        config=config
    )

    return {
        "answer": result["answer"],
        "conversation_id": request.conversation_id
    }
