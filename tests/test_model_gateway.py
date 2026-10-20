from unittest.mock import Mock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from app.services.model_gateway import ModelGateway


INVALID_INPUTS = [
    {"prompt": ""},
    {"prompt": " \n\t"},
    {"prompt": "x" * 10001},
    {"prompt": None},
    {"prompt": 123},
    {"max_new_tokens": 0},
    {"max_new_tokens": -1},
    {"max_new_tokens": 257},
    {"max_new_tokens": 1.5},
    {"max_new_tokens": True},
    {"max_new_tokens": "20"},
    {"temperature": 0},
    {"temperature": -0.1},
    {"temperature": 2.1},
    {"temperature": True},
    {"temperature": "0.8"},
]


@pytest.fixture
def gateway_components():
    with (
        patch("app.services.model_gateway.AutoTokenizer.from_pretrained") as load_tokenizer,
        patch("app.services.model_gateway.AutoModelForCausalLM.from_pretrained") as load_model,
    ):
        tokenizer = load_tokenizer.return_value
        model = load_model.return_value
        tokenizer.return_value.to.return_value = {"input_ids": torch.tensor([[1, 2]])}
        tokenizer.eos_token_id = 0
        tokenizer.decode.return_value = "continuation"
        model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        yield ModelGateway(), tokenizer, model, load_tokenizer, load_model


def test_gateway_loads_once_and_returns_only_continuation(gateway_components):
    gateway, tokenizer, model, load_tokenizer, load_model = gateway_components

    assert gateway.generate("Hello", max_new_tokens=2) == "continuation"
    assert gateway.generate("Hello", max_new_tokens=2) == "continuation"

    load_tokenizer.assert_called_once()
    load_model.assert_called_once()
    assert model.generate.call_count == 2
    assert tokenizer.decode.call_args.args[0].tolist() == [3, 4]
    assert model.generate.call_args.kwargs["max_new_tokens"] == 2


def test_omitted_token_limit_uses_gateway_default(gateway_components):
    gateway, _, model, _, _ = gateway_components

    gateway.generate("Hello")

    assert model.generate.call_args.kwargs["max_new_tokens"] == gateway.max_new_tokens


@pytest.mark.parametrize("invalid", INVALID_INPUTS)
def test_invalid_generation_input_never_reaches_model(gateway_components, invalid):
    gateway, tokenizer, model, _, _ = gateway_components

    with pytest.raises(ValueError):
        gateway.generate(**{"prompt": "Hello", **invalid})

    tokenizer.assert_not_called()
    model.generate.assert_not_called()


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_temperature_is_rejected(gateway_components, temperature):
    gateway, tokenizer, model, _, _ = gateway_components

    with pytest.raises(ValueError):
        gateway.generate("Hello", temperature=temperature)

    tokenizer.assert_not_called()
    model.generate.assert_not_called()


def test_app_reuses_startup_gateway():
    from app import main

    with patch.object(main, "ModelGateway") as constructor:
        assert main.get_model_gateway() is main.model_gateway
        assert main.get_model_gateway() is main.model_gateway

    constructor.assert_not_called()


def test_generate_endpoint_returns_text_and_reuses_gateway():
    from app import main

    with (
        patch.object(main.model_gateway, "generate", return_value="continuation") as generate,
        TestClient(main.app) as client,
    ):
        for _ in range(2):
            response = client.post("/generate", json={"prompt": "Hello"})
            assert response.status_code == 200
            assert response.json() == {"generated_text": "continuation"}

    assert generate.call_count == 2
    generate.assert_called_with(prompt="Hello", max_new_tokens=100, temperature=0.8)


@pytest.mark.parametrize("invalid", INVALID_INPUTS)
def test_generate_endpoint_rejects_invalid_input(invalid):
    from app import main

    with (
        patch.object(main, "get_model_gateway") as get_gateway,
        TestClient(main.app) as client,
    ):
        response = client.post("/generate", json={"prompt": "Hello", **invalid})

    assert response.status_code == 422
    get_gateway.assert_not_called()


def test_chat_calls_reuse_provider_and_preserve_native_tools(gateway_components):
    gateway, tokenizer, model, _, _ = gateway_components
    messages = [{"role": "user", "content": "Calculate 2 + 3"}]
    tools = [{"type": "function", "function": {"name": "calculator"}}]

    with patch("app.services.model_gateway.OllamaProvider") as provider_type:
        expected = provider_type.return_value.generate.return_value
        for _ in range(2):
            result = gateway.generate(model="llama", messages=messages, tools=tools)
            assert result is expected

        provider_type.assert_called_once_with(model="llama3.2:3b")
        assert provider_type.return_value.generate.call_count == 2
        provider_type.return_value.generate.assert_called_with(messages=messages, tools=tools)

    tokenizer.assert_not_called()
    model.generate.assert_not_called()


@pytest.mark.parametrize("arguments", [
    {"prompt": "Hello", "messages": [{"role": "user", "content": "Hello"}]},
    {"model": "unknown", "messages": [{"role": "user", "content": "Hello"}]},
    {"messages": []},
    {"prompt": "Hello", "tools": []},
])
def test_invalid_gateway_api_combinations_are_rejected(gateway_components, arguments):
    gateway, tokenizer, model, _, _ = gateway_components
    with patch("app.services.model_gateway.OllamaProvider") as provider_type:
        with pytest.raises(ValueError):
            gateway.generate(**arguments)
        provider_type.assert_not_called()

    tokenizer.assert_not_called()
    model.generate.assert_not_called()