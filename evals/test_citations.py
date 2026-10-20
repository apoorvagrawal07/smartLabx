import pytest

from app.rag.answer_generator import generate_answer
from evals.citation_checks import validate_citations


@pytest.mark.parametrize("answer, valid_ids, expected_invalid", [
    ("Retries use backoff [S1].", {"S1", "S2"}, []),
    ("Both documents agree [S1] [S2].", {"S1", "S2"}, []),
    ("An unsupported source [S99].", {"S1"}, ["S99"]),
    ("Mixed sources [S1] [S99] [S99].", {"S1"}, ["S99"]),
    ("A citation without evidence [S1].", set(), ["S1"]),
])
def test_citation_ids_belong_to_supplied_sources(answer, valid_ids, expected_invalid):
    assert validate_citations(answer, valid_ids) == expected_invalid


@pytest.mark.parametrize("citation, expected_invalid", [
    ("S1", []),
    ("S99", ["S99"]),
])
def test_generated_answer_citations_match_returned_sources(citation, expected_invalid):
    result = generate_answer(
        "How should transient failures be retried?",
        [{"source": "reliability-guide.md", "content": "Use exponential backoff."}],
        lambda prompt: f"Use exponential backoff [{citation}].",
    )

    valid_ids = {source["id"] for source in result["sources"]}

    assert validate_citations(result["answer"], valid_ids) == expected_invalid