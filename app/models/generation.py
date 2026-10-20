from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: str = Field(min_length=1, max_length=10000)
    max_new_tokens: int = Field(default=100, ge=1, le=256)
    temperature: float = Field(default=0.8, gt=0, le=2.0, allow_inf_nan=False)

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only")
        return value