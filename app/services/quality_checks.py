"""Image quality checks and aggregate scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import ceil
from typing import TYPE_CHECKING, Any

from app.services.body_metrics import BodyRegionStatus
from app.services.feature_engineering import AggregatedVisualFeatures

if TYPE_CHECKING:
    import numpy as np
    from app.core.config import Settings
else:
    np = Any


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """Summary of quality heuristics for a single image."""

    quality_score: float
    warnings: list[str]
    is_blurry: bool


def assess_image_quality(image_bgr: np.ndarray, settings: Settings) -> QualityAssessment:
    """Assess blur, brightness, and resolution using lightweight heuristics."""

    import cv2
    import numpy as np

    height, width = image_bgr.shape[:2]
    warnings: list[str] = []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))

    min_dimension = min(width, height)
    resolution_score = min(1.0, min_dimension / float(settings.min_image_dimension_px * 2))
    blur_score = min(1.0, blur_variance / float(settings.blur_variance_threshold * 2))
    brightness_score = 1.0

    is_blurry = blur_variance < settings.blur_variance_threshold
    if is_blurry:
        warnings.append("Image appears blurry; pose estimates may be less reliable.")

    if min_dimension < settings.min_image_dimension_px:
        warnings.append("Image resolution is low; higher resolution improves landmark stability.")
        resolution_score *= 0.6

    if mean_brightness < settings.dark_brightness_threshold:
        warnings.append("Image is dark; better lighting may improve pose detection.")
        brightness_score = 0.6
    elif mean_brightness > settings.bright_brightness_threshold:
        warnings.append("Image is very bright; avoid overexposed lighting if possible.")
        brightness_score = 0.75

    quality_score = round((resolution_score * 0.35) + (blur_score * 0.45) + (brightness_score * 0.20), 3)
    return QualityAssessment(
        quality_score=max(0.0, min(1.0, quality_score)),
        warnings=warnings,
        is_blurry=is_blurry,
    )


def calculate_overall_quality_score(scores: list[float]) -> float | None:
    """Compute the average quality score across valid images."""

    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)


def build_recommendations(
    *,
    has_successful_detection: bool,
    has_blurry_image: bool,
    has_invalid_image: bool,
    missing_bmi_inputs: bool,
    overall_quality_score: float | None,
    has_missing_torso_metrics: bool,
    missing_body_fat_inputs: bool,
    low_body_fat_confidence: bool,
) -> list[str]:
    """Generate user-facing recommendations from aggregate analysis outcomes."""

    recommendations: list[str] = []
    if not has_successful_detection:
        recommendations.append("Capture at least one image where the subject is clearly visible and centered.")
    if has_blurry_image or (overall_quality_score is not None and overall_quality_score < 0.6):
        recommendations.append("Retake photos with steadier framing, sharper focus, and even lighting.")
    if has_missing_torso_metrics:
        recommendations.append("Keep shoulders and hips fully visible to improve torso-based estimates.")
    if has_invalid_image:
        recommendations.append("Upload JPEG, PNG, or WEBP files that decode correctly on the server.")
    if missing_bmi_inputs:
        recommendations.append("Provide both height_cm and weight_kg if BMI should be returned.")
    if missing_body_fat_inputs:
        recommendations.append("Provide sex, age, height_cm, and weight_kg to enable body fat estimation.")
    if low_body_fat_confidence:
        recommendations.append("Use at least one sharp full-body photo to improve body fat estimation confidence.")
    return recommendations


def build_analysis_feedback(
    *,
    has_successful_detection: bool,
    overall_quality_score: float | None,
    confidence_score: float | None,
    estimated_body_fat_percent: float | None,
    analyzed_regions_summary: list[BodyRegionStatus],
) -> str:
    """Generate a short, readable summary of what looked usable in the analysis."""

    if not has_successful_detection:
        return "No reliable body was detected. Retake at least one clear, centered photo."

    strong_regions = [region.label for region in analyzed_regions_summary if region.taken_into_account]
    weak_regions = [region.label for region in analyzed_regions_summary if not region.taken_into_account]

    if confidence_score is not None and confidence_score >= 0.75:
        opening = "Good analysis base."
    elif confidence_score is not None and confidence_score >= 0.55:
        opening = "Usable analysis, but some signals remain weak."
    else:
        opening = "Low-confidence analysis."

    if overall_quality_score is not None and overall_quality_score >= 0.75:
        quality_note = "Image quality looks solid overall."
    elif overall_quality_score is not None and overall_quality_score >= 0.55:
        quality_note = "Image quality is acceptable but not ideal."
    else:
        quality_note = "Image quality is limiting the analysis."

    if estimated_body_fat_percent is None:
        estimate_note = "A complete body fat estimate could not be produced."
    else:
        estimate_note = "The body fat estimate should still be treated as directional only."

    region_notes: list[str] = []
    if strong_regions:
        region_notes.append(f"Reliable zones: {_join_labels(strong_regions[:4])}.")
    if weak_regions:
        region_notes.append(f"Weak or missing zones: {_join_labels(weak_regions[:4])}.")

    return " ".join([opening, quality_note, *region_notes, estimate_note]).strip()


def build_coaching_feedback(
    *,
    sex: str | None,
    confidence_score: float | None,
    estimated_body_fat_percent: float | None,
    estimated_lean_mass_kg: float | None,
    weight_kg: float | None,
    visual_features: AggregatedVisualFeatures,
    reference_date: date | None = None,
) -> str:
    """Generate a more narrative, coaching-style summary for the frontend."""

    if estimated_body_fat_percent is None or weight_kg is None or estimated_lean_mass_kg is None:
        return (
            "The current data is not strong enough for a coaching-style body fat projection yet. "
            "Use clearer full-body photos and provide age, sex, height_cm, and weight_kg."
        )

    if confidence_score is not None and confidence_score < 0.45:
        return (
            "This estimate is still too low-confidence for a useful coaching projection. "
            "Retake sharper full-body photos before relying on a timeline."
        )

    composition_note = _build_composition_note(
        sex=sex,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
    )
    trajectory_note = _build_trajectory_note(
        sex=sex,
        estimated_body_fat_percent=estimated_body_fat_percent,
        estimated_lean_mass_kg=estimated_lean_mass_kg,
        weight_kg=weight_kg,
        reference_date=reference_date,
    )

    confidence_note = ""
    if confidence_score is not None and confidence_score < 0.65:
        confidence_note = " Confidence is moderate, so treat this as directional only."

    return f"{composition_note} {trajectory_note}{confidence_note}".strip()


def _join_labels(labels: list[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _build_composition_note(
    *,
    sex: str | None,
    estimated_body_fat_percent: float,
    visual_features: AggregatedVisualFeatures,
) -> str:
    waist_to_hip = visual_features.estimated_waist_to_hip_ratio
    waist_to_bbox = visual_features.estimated_waist_to_bbox_height_ratio
    canonical_sex = (sex or "").lower()

    if canonical_sex == "male":
        if estimated_body_fat_percent <= 13.5:
            return "Overall composition looks fairly lean."
        if (
            waist_to_hip is not None and waist_to_hip >= 0.92
        ) or (
            waist_to_bbox is not None and waist_to_bbox >= 0.205
        ):
            return "Good overall composition. Mild abdominal fat retention is still likely."
        if estimated_body_fat_percent <= 17.0:
            return "Good overall composition. You look moderately lean already."
        return "General composition looks usable, but body fat still appears moderate."

    if canonical_sex == "female":
        if estimated_body_fat_percent <= 21.0:
            return "Overall composition looks fairly lean."
        if (
            waist_to_hip is not None and waist_to_hip >= 0.82
        ) or (
            waist_to_bbox is not None and waist_to_bbox >= 0.235
        ):
            return "Good overall composition. Mild fat retention around the waist is still likely."
        if estimated_body_fat_percent <= 28.0:
            return "Good overall composition. You already look moderately lean."
        return "General composition looks usable, but body fat still appears moderate."

    return "General composition looks usable from the current estimate."


def _build_trajectory_note(
    *,
    sex: str | None,
    estimated_body_fat_percent: float,
    estimated_lean_mass_kg: float,
    weight_kg: float,
    reference_date: date | None,
) -> str:
    target_body_fat = _target_body_fat(sex)
    if target_body_fat is None:
        return "No coaching projection is available because sex is missing."
    if estimated_body_fat_percent <= target_body_fat:
        return f"You are already around the illustrative target of {target_body_fat:.0f}% body fat."

    target_weight = estimated_lean_mass_kg / (1.0 - (target_body_fat / 100.0))
    weight_to_lose = max(0.0, weight_kg - target_weight)
    if weight_to_lose <= 0.2:
        return f"You are already very close to an illustrative target near {target_body_fat:.0f}% body fat."

    weekly_rate_kg = 0.5
    weeks_needed = weight_to_lose / weekly_rate_kg
    today = reference_date or date.today()
    projected_date = today + timedelta(days=ceil(weeks_needed * 7))
    month_label = projected_date.strftime("%B %Y")
    return (
        f"At an illustrative pace of about {weekly_rate_kg:.1f} kg per week, "
        f"you could approach {target_body_fat:.0f}% estimated body fat around {month_label}."
    )


def _target_body_fat(sex: str | None) -> float | None:
    canonical_sex = (sex or "").lower()
    if canonical_sex == "male":
        return 15.0
    if canonical_sex == "female":
        return 24.0
    return None
