"""Request schemas for the analysis endpoint."""

from typing import Annotated

from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalyzeRequestMetadata(BaseModel):
    """Validated form fields submitted alongside uploaded images."""

    sex: str | None = Field(default=None, max_length=64)
    gender: str | None = Field(default=None, max_length=64)
    age: int | None = Field(default=None, ge=0, le=120)
    height_cm: float | None = Field(default=None, gt=0, le=300)
    weight_kg: float | None = Field(default=None, gt=0, le=500)
    user_id: str | None = Field(default=None, max_length=128)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sex": "female",
                "gender": "female",
                "age": 31,
                "height_cm": 168.0,
                "weight_kg": 63.5,
                "user_id": "user_12345",
            }
        }
    )

    @field_validator("sex", "gender", "user_id", mode="before")
    @classmethod
    def normalize_optional_string(cls, value: str | None) -> str | None:
        """Trim empty string inputs from multipart form values."""

        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @classmethod
    def as_form(
        cls,
        sex: Annotated[str | None, Form()] = None,
        gender: Annotated[str | None, Form()] = None,
        age: Annotated[int | None, Form()] = None,
        height_cm: Annotated[float | None, Form()] = None,
        weight_kg: Annotated[float | None, Form()] = None,
        user_id: Annotated[str | None, Form()] = None,
    ) -> "AnalyzeRequestMetadata":
        """Build the request metadata model from multipart form fields."""

        return cls(
            sex=sex,
            gender=gender,
            age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            user_id=user_id,
        )

    @property
    def resolved_sex(self) -> str | None:
        """Return the canonical sex value, preferring `sex` over legacy `gender`."""

        return self.sex or self.gender
