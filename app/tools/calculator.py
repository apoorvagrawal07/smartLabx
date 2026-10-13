import re

from simpleeval import simple_eval


def _extract_expression(expression: str) -> str:
    text = expression.strip()

    text = re.sub(
        r"(?i)^\s*(?:calculate|compute|what\s+is|what\s+is\s+the\s+value\s+of)\s+",
        "",
        text,
    )

    candidates = re.findall(r"[\d\s+\-*/().]+", text)
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned and re.search(r"\d", cleaned) and re.search(r"[+\-*/]", cleaned):
            return cleaned

    cleaned = re.sub(r"[^0-9+\-*/().\s]", "", text)
    cleaned = cleaned.strip()
    if cleaned and re.search(r"\d", cleaned) and re.search(r"[+\-*/]", cleaned):
        return cleaned

    return text


def calculate(expression: str) -> str:

    try:
        normalized_expression = _extract_expression(expression)
        return str(simple_eval(normalized_expression))

    except Exception:
        return "Unable to calculate expression."
