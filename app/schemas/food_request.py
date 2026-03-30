"""Request schemas for food analysis."""

from typing import Annotated

from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalyzeFoodRequestMetadata(BaseModel):
    """Validated form fields submitted alongside food images."""

    user_id: str | None = Field(default=None, max_length=128)
    meal_context: str | None = Field(default=None, max_length=64)
    locale: str | None = Field(default="fr-FR", max_length=16)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "meal_context": "lunch",
                "locale": "fr-FR",
            }
        }
    )

    @field_validator("user_id", "meal_context", "locale", mode="before")
    @classmethod
    def normalize_optional_string(cls, value: str | None) -> str | None:
        """Trim empty multipart values to None."""

        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @classmethod
    def as_form(
        cls,
        user_id: Annotated[str | None, Form()] = None,
        meal_context: Annotated[str | None, Form()] = None,
        locale: Annotated[str | None, Form()] = "fr-FR",
    ) -> "AnalyzeFoodRequestMetadata":
        """Build the request metadata model from multipart form fields."""

        return cls(
            user_id=user_id,
            meal_context=meal_context,
            locale=locale,
        )
