
import re


CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def validate_citations(
    answer: str,
    valid_source_ids: set[str],
) -> list[str]:
    cited_ids = set(CITATION_PATTERN.findall(answer))
    return sorted(cited_ids - valid_source_ids)


def test_valid_citations():
    answer = "Retries use exponential backoff [S1]."

    invalid = validate_citations(
        answer,
        {"S1", "S2"},
    )

    assert invalid == []


def test_unknown_citation_is_rejected():
    answer = "The service retries five times [S99]."

    invalid = validate_citations(answer, {"S1"})

    assert invalid == ["S99"]
