from unittest.mock import Mock, call, patch

import pytest

from app.agents.permissions import ToolPermissionError
from app.rag.retry import retry_transient
from app.services.structured_output_service import StructuredOutputError


@pytest.mark.parametrize("attempts", [1, 2, 3])
@pytest.mark.parametrize("error_type", [TimeoutError, ConnectionError])
def test_transient_retries_stop_at_attempt_limit(attempts, error_type):
    failure = error_type("Temporary failure")
    operation = Mock(side_effect=failure)

    with patch("app.rag.retry.time.sleep") as sleep:
        with pytest.raises(error_type) as error:
            retry_transient(operation, attempts=attempts, base_delay=0.25)

    assert error.value is failure
    assert operation.call_count == attempts
    assert sleep.call_args_list == [
        call(0.25 * (2 ** attempt)) for attempt in range(attempts - 1)
    ]


def test_default_retry_budget_is_three_attempts():
    operation = Mock(side_effect=TimeoutError("Temporary failure"))

    with patch("app.rag.retry.time.sleep") as sleep:
        with pytest.raises(TimeoutError):
            retry_transient(operation)

    assert operation.call_count == 3
    assert sleep.call_args_list == [call(0.25), call(0.5)]


@pytest.mark.parametrize("failures", [0, 2])
def test_retry_stops_on_success_and_returns_original_value(failures):
    expected = object()
    operation = Mock(side_effect=[TimeoutError("Temporary failure")] * failures + [expected])

    with patch("app.rag.retry.time.sleep") as sleep:
        result = retry_transient(operation, attempts=4)

    assert result is expected
    assert operation.call_count == failures + 1
    assert sleep.call_count == failures


@pytest.mark.parametrize("failure", [
    ValueError("Invalid input"),
    ToolPermissionError("Tool access denied"),
    StructuredOutputError("Invalid model output"),
])
def test_non_transient_failures_are_not_retried(failure):
    operation = Mock(side_effect=failure)

    with patch("app.rag.retry.time.sleep") as sleep:
        with pytest.raises(type(failure)) as error:
            retry_transient(operation)

    assert error.value is failure
    operation.assert_called_once_with()
    sleep.assert_not_called()


@pytest.mark.parametrize("attempts", [0, -1])
def test_invalid_attempt_budget_prevents_execution(attempts):
    operation = Mock()

    with patch("app.rag.retry.time.sleep") as sleep:
        with pytest.raises(ValueError, match="attempts must be at least 1"):
            retry_transient(operation, attempts=attempts)

    operation.assert_not_called()
    sleep.assert_not_called()


@pytest.mark.parametrize("base_delay", [-1, float("nan"), float("inf")])
def test_invalid_delay_prevents_execution(base_delay):
    operation = Mock()
    with pytest.raises(ValueError, match="base_delay"):
        retry_transient(operation, base_delay=base_delay)
    operation.assert_not_called()