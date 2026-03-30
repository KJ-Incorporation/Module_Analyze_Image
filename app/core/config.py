"""Application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_name: str = "Weighty Vision Module"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    max_images_per_request: int = Field(default=5, ge=1, le=20)
    max_file_size_mb: int = Field(default=10, ge=1, le=50)
    min_image_dimension_px: int = Field(default=256, ge=64)
    blur_variance_threshold: float = Field(default=90.0, gt=0.0)
    dark_brightness_threshold: float = Field(default=45.0, ge=0.0, le=255.0)
    bright_brightness_threshold: float = Field(default=220.0, ge=0.0, le=255.0)
    pose_model_path: str = "models/pose_landmarker_lite.task"
    pose_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    pose_presence_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    pose_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metric_visibility_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    body_fat_model_version: str = "heuristic-bodyfat-v0.3.0"
    food_model_version: str = "heuristic-food-v0.2.0"

    model_config = SettingsConfigDict(
        env_prefix="WEIGHTY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
