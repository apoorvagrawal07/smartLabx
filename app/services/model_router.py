class ModelRouter:

    ROUTES = {
        "fast": "phi",
        "reasoning": "llama"
    }

    def select_model(self, mode: str) -> str:

        model = self.ROUTES.get(mode)

        if model is None:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        return model