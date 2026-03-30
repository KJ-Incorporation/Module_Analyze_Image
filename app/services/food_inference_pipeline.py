"""End-to-end inference pipeline for food analysis."""

from __future__ import annotations

import logging

from fastapi import UploadFile

from app.core.config import Settings
from app.schemas.food_request import AnalyzeFoodRequestMetadata
from app.schemas.food_response import (
    AnalyzeFoodResponse,
    FoodAnalysisNotesResponse,
    FoodDetectedItemResponse,
    FoodImageAnalysisResponse,
    FoodMealFeedbackResponse,
)
from app.services.food_estimator import (
    build_food_analysis_notes,
    build_food_meal_feedback,
    build_food_recommendations,
    estimate_food_from_features,
    merge_food_estimates,
)
from app.services.food_feature_engineering import extract_food_visual_features
from app.services.image_loader import (
    ImageLoadError,
    UnsupportedImageFormatError,
    load_image_from_upload,
)
from app.services.quality_checks import assess_image_quality

LOGGER = logging.getLogger(__name__)


class FoodInferencePipeline:
    """Application service that orchestrates multi-image food analysis."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def analyze(
        self,
        *,
        metadata: AnalyzeFoodRequestMetadata,
        images: list[UploadFile],
    ) -> AnalyzeFoodResponse:
        """Analyze food images and return a consolidated JSON payload."""

        results: list[FoodImageAnalysisResponse] = []
        per_image_estimates = []

        for upload in images:
            LOGGER.info("Analyzing uploaded food image '%s'", upload.filename)
            try:
                loaded_image = await load_image_from_upload(upload, self._settings.max_file_size_mb)
            except UnsupportedImageFormatError as exc:
                results.append(
                    _build_failed_food_response(
                        upload=upload,
                        processing_status="unsupported_format",
                        warning=str(exc),
                    )
                )
                continue
            except ImageLoadError as exc:
                results.append(
                    _build_failed_food_response(
                        upload=upload,
                        processing_status="invalid_image",
                        warning=str(exc),
                    )
                )
                continue

            quality = assess_image_quality(loaded_image.image_bgr, self._settings)
            features = extract_food_visual_features(loaded_image.image_bgr)
            estimate = estimate_food_from_features(
                features=features,
                quality_score=quality.quality_score,
            )
            per_image_estimates.append(estimate)

            results.append(
                FoodImageAnalysisResponse(
                    filename=loaded_image.filename,
                    content_type=loaded_image.content_type,
                    processing_status="success" if estimate.food_detected else "no_food_detected",
                    food_detected=estimate.food_detected,
                    confidence_score=estimate.confidence_score,
                    quality_score=quality.quality_score,
                    image_width=loaded_image.width,
                    image_height=loaded_image.height,
                    meal_label=estimate.meal_label,
                    estimated_health_profile=estimate.estimated_health_profile,
                    detected_items=[
                        FoodDetectedItemResponse(label=item.label, confidence=item.confidence)
                        for item in estimate.detected_items
                    ],
                    warnings=list(dict.fromkeys([*quality.warnings, *estimate.warnings])),
                )
            )

        merged = merge_food_estimates(per_image_estimates)
        attention, parfait, progression = build_food_analysis_notes(merged)
        meal_attention, meal_parfait, meal_progression = build_food_meal_feedback(merged)
        recommendations = build_food_recommendations(merged)

        return AnalyzeFoodResponse(
            user_id=metadata.user_id,
            meal_context=metadata.meal_context,
            locale=metadata.locale,
            images_processed=len(results),
            food_detected=merged.food_detected,
            meal_label=merged.meal_label,
            detected_items=[
                FoodDetectedItemResponse(label=item.label, confidence=item.confidence)
                for item in merged.detected_items
            ],
            estimated_portion_label=merged.estimated_portion_label,
            estimated_portion_confidence=merged.estimated_portion_confidence,
            estimated_calories_kcal=merged.estimated_calories_kcal,
            estimated_protein_g=merged.estimated_protein_g,
            estimated_carbs_g=merged.estimated_carbs_g,
            estimated_fat_g=merged.estimated_fat_g,
            estimated_health_profile=merged.estimated_health_profile,
            confidence_score=merged.confidence_score,
            model_version=self._settings.food_model_version,
            analysis_notes=FoodAnalysisNotesResponse(
                attention=attention,
                parfait=parfait,
                progression=progression,
            ),
            meal_feedback=FoodMealFeedbackResponse(
                attention=meal_attention,
                parfait=meal_parfait,
                progression=meal_progression,
            ),
            warnings=list(dict.fromkeys([warning for result in results for warning in result.warnings])),
            recommendations=recommendations,
            results=results,
        )


def _build_failed_food_response(
    *,
    upload: UploadFile,
    processing_status: str,
    warning: str,
) -> FoodImageAnalysisResponse:
    return FoodImageAnalysisResponse(
        filename=upload.filename or "uploaded_image",
        content_type=upload.content_type,
        processing_status=processing_status,
        food_detected=False,
        confidence_score=0.0,
        quality_score=None,
        image_width=None,
        image_height=None,
        meal_label=None,
        estimated_health_profile=None,
        detected_items=[],
        warnings=[warning],
    )
