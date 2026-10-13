import time
import uuid

from fastapi import Request


async def request_context_middleware(request: Request, call_next):

    request_id = str(uuid.uuid4())

    start_time = time.perf_counter()

    request.state.request_id = request_id

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(duration)

    print(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"duration={duration:.4f}s"
    )

    return response