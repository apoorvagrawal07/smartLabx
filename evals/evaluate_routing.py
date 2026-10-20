
import json
from pathlib import Path

from app.agents.router import AgentRouter


DATASET = Path(__file__).with_name("dataset.jsonl")


def predict_route(message: str) -> str:
    """Adapt your real router's output to a route name."""
    return AgentRouter().decide(message).intent


def main():
    cases = [
        json.loads(line)
        for line in DATASET.read_text().splitlines()
        if line.strip()
    ]

    correct = 0
    failures = []

    for case in cases:
        expected = case["expected_route"]

        try:
            predicted = predict_route(case["input"])
            passed = predicted == expected
        except Exception as exc:
            predicted = f"ERROR: {type(exc).__name__}: {exc}"
            passed = False

        correct += int(passed)

        print(
            f"{'PASS' if passed else 'FAIL'} | "
            f"expected={expected} | predicted={predicted} | "
            f"input={case['input']}"
        )

        if not passed:
            failures.append(case["input"])

    total = len(cases)
    accuracy = correct / total if total else 0.0

    print(f"\nRouting accuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"Failed cases: {len(failures)}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
