
import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    pass


def parse_structured_output(
    raw_output: str,
    schema: type[T],
) -> T:
    try:
        data = json.loads(raw_output)
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise StructuredOutputError(
            f"Model returned invalid {schema.__name__} output"
        ) from exc
