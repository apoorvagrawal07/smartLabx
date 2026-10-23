from http import HTTPStatus
from contextlib import asynccontextmanager
import os
from threading import Lock

from fastapi import FastAPI, Path, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field

from pydantic import BaseModel, Field, field_validator
from app.middleware.request_context import (
    request_context_middleware
)

from app.graph.workflow import build_graph
from app.rag.rag_service import RAGService
from app.rag.retriever import HybridRetriever
from app.rag.reranker import Reranker
from app.rag.retrieval_pipeline import RetrievalPipeline
from app.services.model_gateway import ModelGateway
from app.tools.tool_executor import ToolExecutor

from functools import lru_cache

from fastapi import HTTPException
from app.models.generation import GenerateRequest
from app.services.conversation_access import SESSION_COOKIE
from app.services.persistence import open_persistence
from app.services.memory_service import MemoryService
from app.models.structured_outputs import MemoryEntry


@asynccontextmanager
async def lifespan(app: FastAPI):
    database_url = os.environ.get("DATABASE_URL")
    if database_url is None:
        yield
        return

    with open_persistence(database_url) as (checkpointer, ownership):
        retriever = LazyRetriever()
        app.state.workflow = build_graph(
            model_gateway=model_gateway,
            rag_service=RAGService(
                retriever=retriever,
                model_gateway=model_gateway,
                include_sources=True,
            ),
            tool_executor=ToolExecutor(),
            checkpointer=checkpointer,
        )
        app.state.conversation_access = ownership
        app.state.memory_service = MemoryService(database_url)
        try:
            yield
        finally:
            retriever.close()
            app.state.workflow = None
            app.state.conversation_access = None
            app.state.memory_service = None


app = FastAPI(lifespan=lifespan)
workflow = None
conversation_access = None


@app.exception_handler(RequestValidationError)
async def invalid_request_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Invalid request"})


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    messages = {
        400: "Invalid request",
        422: "Invalid request",
        500: "Internal server error",
        503: "Model inference is temporarily unavailable",
    }
    try:
        detail = messages.get(exc.status_code, HTTPStatus(exc.status_code).phrase)
    except ValueError:
        detail = "Request failed"
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError):
    return JSONResponse(status_code=403, content={"detail": "Forbidden"})


@lru_cache(maxsize=1)
def get_model_gateway():
    return ModelGateway()


@app.post("/generate")
def generate_text(request: GenerateRequest):
    try:
        gateway = get_model_gateway()
        result = gateway.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        return {"generated_text": result}

    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid request") from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model inference is temporarily unavailable",
        ) from exc


app.middleware("http")(
    request_context_middleware
)

model_gateway = get_model_gateway()


class LazyRetriever:

    def __init__(self):
        self._retriever = None
        self._lock = Lock()

    def retrieve(self, query, limit=3):
        with self._lock:
            if self._retriever is None:
                reranker = Reranker()
                self._retriever = RetrievalPipeline(HybridRetriever(), reranker)

        return self._retriever.retrieve(query, candidate_count=max(20, limit), final_count=limit)

    def close(self):
        with self._lock:
            if self._retriever is not None:
                self._retriever.hybrid_retriever.vector_store.connection.close()
                self._retriever = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    conversation_id: str = Field(min_length=1, max_length=128)

    @field_validator("message", "conversation_id")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be blank")
        return value


def memory_context(request: Request):
    service = getattr(request.app.state, "memory_service", None)
    ownership = conversation_access if conversation_access is not None else getattr(
        request.app.state, "conversation_access", None
    )
    if service is None or ownership is None:
        raise HTTPException(status_code=503, detail="Persistent storage is unavailable")
    owner_id = ownership.owner_id(request.cookies.get(SESSION_COOKIE))
    return service, owner_id


@app.get("/memories", response_model=list[MemoryEntry])
def list_memories(request: Request):
    service, owner_id = memory_context(request)
    return service.get_all(owner_id)


@app.post("/memories", response_model=MemoryEntry)
def save_memory(entry: MemoryEntry, request: Request):
    service, owner_id = memory_context(request)
    service.save(owner_id, entry.key, entry.value)
    return entry


@app.delete("/memories/{key}", status_code=204)
def delete_memory(
    request: Request,
    key: str = Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$"),
):
    service, owner_id = memory_context(request)
    service.delete(owner_id, key)
    return Response(status_code=204)


class ChatResponse(BaseModel):

    answer: str
    conversation_id: str
    sources: list[dict] = Field(default_factory=list)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request):
    """Continue a conversation using its owner's session cookie.

    Clients must retain cookies. Ownership and checkpoints persist in PostgreSQL.
    Requests for the same conversation should be submitted sequentially.
    """

    active_workflow = workflow if workflow is not None else getattr(http_request.app.state, "workflow", None)
    ownership = conversation_access if conversation_access is not None else getattr(
        http_request.app.state, "conversation_access", None
    )
    if active_workflow is None or ownership is None:
        raise HTTPException(status_code=503, detail="Persistent storage is unavailable")

    http_request.state.conversation_session = ownership.authorize(
        request.conversation_id,
        http_request.cookies.get(SESSION_COOKIE),
    )

    memory_service = getattr(http_request.app.state, "memory_service", None)
    memories = []
    if memory_service is not None:
        memories = memory_service.get_all(
            ownership.owner_id(http_request.state.conversation_session), limit=20
        )

    config = {
        "configurable": {
            "thread_id": request.conversation_id
        }
    }

    checkpoint = active_workflow.get_state(config)
    history = checkpoint.values.get("messages", [])

    result = active_workflow.invoke(
        {
            "messages": history + [
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            "user_message": request.message,
            "user_memories": memories,
            "iteration_count": 0,
            "sources": [],
        },
        config=config
    )

    return {
        "answer": result["answer"],
        "conversation_id": request.conversation_id,
        "sources": result.get("sources", []),
    }
