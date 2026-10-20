
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

logger = logging.getLogger("ai_platform")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

request_id_context = ContextVar("request_id", default=None)

LOG_FIELDS = {
    "request_id", "method", "path", "status_code", "duration_ms",
    "operation", "status", "error_type", "backend", "model",
}


def log_event(event: str, **fields):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **{key: value for key, value in fields.items() if key in LOG_FIELDS},
    }
    request_id = request_id_context.get()
    if request_id is not None:
        record.setdefault("request_id", request_id)
    logger.info(json.dumps(record))


@contextmanager
def trace_operation(operation: str, **fields):
    start = time.perf_counter()
    status = "success"

    try:
        yield
    except Exception as exc:
        status = "error"
        log_event(
            "operation_failed",
            operation=operation,
            error_type=type(exc).__name__,
            **fields,
        )
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_event(
            "operation_completed",
            operation=operation,
            status=status,
            duration_ms=duration_ms,
            **fields,
        )
