import unittest

from app.providers.ollama_provider import OllamaProvider
from app.tools.calculator import calculate


class OllamaProviderTests(unittest.TestCase):
    def test_provider_initializes_client(self):
        provider = OllamaProvider("llama3.2:3b")

        self.assertTrue(hasattr(provider, "client"))
        self.assertTrue(callable(getattr(provider.client, "chat", None)))

    def test_calculator_extracts_expression_from_natural_language(self):
        self.assertEqual(calculate("what is 135*7?"), "945")


if __name__ == "__main__":
    unittest.main()
