import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
from httpx import ConnectError, ReadTimeout
from ollama import ResponseError

from app.providers.ollama_provider import OllamaProvider
from app.tools.calculator import calculate


class OllamaProviderTests(unittest.TestCase):
    def test_provider_initializes_client(self):
        provider = OllamaProvider("llama3.2:3b")

        self.assertTrue(hasattr(provider, "client"))
        self.assertTrue(callable(getattr(provider.client, "chat", None)))

    def test_calculator_extracts_expression_from_natural_language(self):
        self.assertEqual(calculate("what is 135*7?"), "945")


@pytest.mark.parametrize("failure", [
    ConnectionError("temporary"), ConnectError("temporary"), ReadTimeout("temporary"),
])
def test_provider_transport_retries_are_bounded(failure):
    with patch("app.providers.ollama_provider.Client") as client_type:
        provider = OllamaProvider("llama3.2:3b")
    provider.client.chat.side_effect = failure
    with patch("app.rag.retry.time.sleep") as sleep:
        with pytest.raises(type(failure)):
            provider.generate([{"role": "user", "content": "hello"}])
    assert provider.client.chat.call_count == 3
    assert sleep.call_args_list == [call(0.25), call(0.5)]
    client_type.assert_called_once_with(host="http://localhost:11434", timeout=60)


def test_provider_stops_retrying_after_success():
    provider = OllamaProvider("llama3.2:3b")
    response = SimpleNamespace(message=SimpleNamespace(content="hello", tool_calls=[]))
    provider.client.chat = Mock(side_effect=[ReadTimeout("temporary"), response])
    with patch("app.rag.retry.time.sleep"):
        assert provider.generate([{"role": "user", "content": "hello"}]).content == "hello"
    assert provider.client.chat.call_count == 2


@pytest.mark.parametrize("failure", [PermissionError("denied"), ResponseError("invalid", status_code=400)])
def test_provider_does_not_retry_non_transient_failures(failure):
    provider = OllamaProvider("llama3.2:3b")
    provider.client.chat = Mock(side_effect=failure)
    with patch("app.rag.retry.time.sleep") as sleep:
        with pytest.raises(type(failure)):
            provider.generate([{"role": "user", "content": "hello"}])
    provider.client.chat.assert_called_once()
    sleep.assert_not_called()


def test_provider_uses_environment_host_and_explicit_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    with patch("app.providers.ollama_provider.Client") as client_type:
        OllamaProvider("llama3.2:3b")
        client_type.assert_called_with(host="http://host.docker.internal:11434", timeout=60)
        OllamaProvider("llama3.2:3b", host="http://explicit:11434")
        client_type.assert_called_with(host="http://explicit:11434", timeout=60)


if __name__ == "__main__":
    unittest.main()
