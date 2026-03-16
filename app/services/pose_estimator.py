"""MediaPipe Pose Landmarker integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from app.core.config import Settings
from app.services.body_metrics import LANDMARK_NAME_BY_INDEX, PixelLandmark

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:  # pragma: no cover - handled at runtime if dependency is absent.
    mp = None
    python = None
    vision = None


class PoseEstimatorInitializationError(RuntimeError):
    """Raised when the MediaPipe pose estimator cannot be created."""


class PoseEstimationError(RuntimeError):
    """Raised when an image cannot be processed by the estimator."""


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-aligned bounding box around the visible body."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class PoseEstimationResult:
    """Pose estimation output for one image."""

    body_detected: bool
    confidence_score: float | None
    bbox: BoundingBox | None
    landmarks: list[PixelLandmark]
    warnings: list[str]


class MediaPipePoseEstimator:
    """Thin wrapper around MediaPipe Pose Landmarker."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._landmarker: Any | None = None
        self._initialization_error: str | None = None
        self._initialize()

    @property
    def is_available(self) -> bool:
        """Whether the underlying landmarker is ready to serve requests."""

        return self._landmarker is not None

    @property
    def initialization_error(self) -> str | None:
        """Human-readable initialization failure reason, if any."""

        return self._initialization_error

    def _initialize(self) -> None:
        if mp is None or python is None or vision is None:
            self._initialization_error = "MediaPipe is not installed in the current environment."
            LOGGER.error(self._initialization_error)
            return

        model_path = self._resolve_model_path(self._settings.pose_model_path)
        if not model_path.exists():
            self._initialization_error = (
                f"Pose Landmarker model not found at '{model_path}'. "
                "Download the .task model and set WEIGHTY_POSE_MODEL_PATH if needed."
            )
            LOGGER.error(self._initialization_error)
            return

        try:
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=self._settings.pose_detection_confidence,
                min_pose_presence_confidence=self._settings.pose_presence_confidence,
                min_tracking_confidence=self._settings.pose_tracking_confidence,
                output_segmentation_masks=False,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - dependent on local runtime.
            self._initialization_error = f"Failed to initialize MediaPipe Pose Landmarker: {exc}"
            LOGGER.exception(self._initialization_error)
            self._landmarker = None

    def _resolve_model_path(self, configured_path: str) -> Path:
        """Resolve the configured model path relative to the project root when needed."""

        path = Path(configured_path)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()

    def estimate_pose(self, image_bgr) -> PoseEstimationResult:
        """Run pose detection against a decoded BGR image."""

        if self._landmarker is None:
            raise PoseEstimatorInitializationError(
                self._initialization_error or "Pose estimator is unavailable."
            )

        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = self._landmarker.detect(mp_image)
        except Exception as exc:  # pragma: no cover - dependent on local runtime.
            raise PoseEstimationError(f"MediaPipe failed to analyze the image: {exc}") from exc

        if not result.pose_landmarks:
            return PoseEstimationResult(
                body_detected=False,
                confidence_score=None,
                bbox=None,
                landmarks=[],
                warnings=["No human body was detected in the image."],
            )

        height, width = image_bgr.shape[:2]
        landmarks = self._to_pixel_landmarks(result.pose_landmarks[0], width=width, height=height)
        bbox = _compute_bounding_box(landmarks, width=width, height=height)
        confidence = _compute_confidence_score(landmarks)

        warnings: list[str] = []
        if bbox is None:
            warnings.append("Body detected, but bounding box estimation is weak.")

        return PoseEstimationResult(
            body_detected=True,
            confidence_score=confidence,
            bbox=bbox,
            landmarks=landmarks,
            warnings=warnings,
        )

    def _to_pixel_landmarks(self, landmarks, *, width: int, height: int) -> list[PixelLandmark]:
        pixel_landmarks: list[PixelLandmark] = []
        for index, landmark in enumerate(landmarks):
            x_px = min(max(landmark.x * width, 0.0), float(width))
            y_px = min(max(landmark.y * height, 0.0), float(height))
            pixel_landmarks.append(
                PixelLandmark(
                    index=index,
                    name=LANDMARK_NAME_BY_INDEX.get(index, f"landmark_{index}"),
                    x_px=round(x_px, 2),
                    y_px=round(y_px, 2),
                    visibility=round(float(getattr(landmark, "visibility", 0.0) or 0.0), 4),
                    presence=round(float(getattr(landmark, "presence", 0.0) or 0.0), 4),
                )
            )
        return pixel_landmarks


def _compute_bounding_box(
    landmarks: list[PixelLandmark],
    *,
    width: int,
    height: int,
    min_visibility: float = 0.3,
) -> BoundingBox | None:
    visible_landmarks = [landmark for landmark in landmarks if (landmark.visibility or 0.0) >= min_visibility]
    usable_landmarks = visible_landmarks or landmarks
    if not usable_landmarks:
        return None

    x_values = [landmark.x_px for landmark in usable_landmarks]
    y_values = [landmark.y_px for landmark in usable_landmarks]
    x_min = max(0, int(min(x_values)))
    y_min = max(0, int(min(y_values)))
    x_max = min(width, int(max(x_values)))
    y_max = min(height, int(max(y_values)))
    return BoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        width=max(0, x_max - x_min),
        height=max(0, y_max - y_min),
    )


def _compute_confidence_score(landmarks: list[PixelLandmark]) -> float | None:
    reliable_scores = []
    for landmark in landmarks:
        visibility = landmark.visibility if landmark.visibility is not None else 0.0
        presence = landmark.presence if landmark.presence is not None else 0.0
        reliable_scores.append((visibility + presence) / 2.0)
    if not reliable_scores:
        return None
    return round(sum(reliable_scores) / len(reliable_scores), 3)
