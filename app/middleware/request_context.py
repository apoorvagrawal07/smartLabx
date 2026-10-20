import time
import uuid

from fastapi import Request
from starlette.responses import JSONResponse

from app.services.observability import log_event, request_id_context
from app.services.conversation_access import SESSION_COOKIE


async def request_context_middleware(request: Request, call_next):

    request_id = str(uuid.uuid4())

    start_time = time.perf_counter()

    request.state.request_id = request_id

    token = request_id_context.set(request_id)
    status_code = 500
    error_type = None
    try:
        try:
            response = await call_next(request)
        except Exception as exc:
            error_type = type(exc).__name__
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )
        status_code = response.status_code
    finally:
        duration = time.perf_counter() - start_time
        fields = {
            "request_id": request_id,
            "method": request.method,
            "path": getattr(request.scope.get("route"), "path", "<unmatched>"),
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2),
        }
        try:
            if status_code >= 400:
                if error_type is not None:
                    fields["error_type"] = error_type
                log_event("http_request_failed", **fields)
            log_event("http_request_completed", **fields)
        finally:
            request_id_context.reset(token)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(duration)

    session_id = getattr(request.state, "conversation_session", None)
    if session_id is not None:
        response.set_cookie(
            SESSION_COOKIE,
            session_id,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
        )

    return response