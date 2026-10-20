
import json

import pytest

from evals.evaluate_routing import DATASET, predict_route


def load_routing_cases():
    cases = []

    for line in DATASET.read_text().splitlines():
        if not line.strip():
            continue

        case = json.loads(line)

        cases.append(case)

    return cases


@pytest.mark.parametrize(
    "case",
    load_routing_cases(),
    ids=lambda case: case["input"],
)
def test_route(case):
    assert predict_route(case["input"]) == case["expected_route"]
