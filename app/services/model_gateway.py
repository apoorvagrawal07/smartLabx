
# app/services/model_gateway.py

from threading import Lock

import torch
from app.services.observability import trace_operation
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.models.generation import GenerateRequest
from app.models.llm import LLMResponse
from app.providers.ollama_provider import OllamaProvider


class ModelGateway:
    CHAT_MODELS = {
        "llama": "llama3.2:3b",
        "phi": "phi4-mini:latest",
    }

    def __init__(
        self,
        model_name: str = "distilbert/distilgpt2",
        max_new_tokens: int = 128,
        timeout_seconds: int = 60,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self._lock = Lock()
        self._chat_providers = {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    @trace_operation("model_inference")
    def generate(
        self,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.8,
        *,
        model: str | None = None,
        messages=None,
        tools=None,
    ) -> str | LLMResponse:
        if messages is not None:
            if prompt is not None:
                raise ValueError("Provide either prompt or messages, not both")
            if not messages:
                raise ValueError("Messages cannot be empty")
            model_key = model or "llama"
            if model_key not in self.CHAT_MODELS:
                raise ValueError(f"Unknown chat model: {model_key}")

            with self._lock:
                if model_key not in self._chat_providers:
                    self._chat_providers[model_key] = OllamaProvider(
                        model=self.CHAT_MODELS[model_key]
                    )
                provider = self._chat_providers[model_key]

            return provider.generate(messages=messages, tools=tools)

        if model is not None or tools is not None:
            raise ValueError("Chat model and tools require messages")

        request = GenerateRequest(
            prompt=prompt,
            max_new_tokens=(
                self.max_new_tokens if max_new_tokens is None else max_new_tokens
            ),
            temperature=temperature,
        )
        token_limit = request.max_new_tokens

        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=900,
        ).to(self.device)

        try:
            # Serialize inference for this simple local implementation.
            with self._lock, torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=token_limit,
                    do_sample=True,
                    temperature=request.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Return only newly generated tokens, not the prompt.
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

        except (RuntimeError, OSError) as exc:
            raise RuntimeError("Local model inference failed") from exc
