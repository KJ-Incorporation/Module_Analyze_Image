"""Feature engineering for multi-image body fat inference."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from app.schemas.response import BoundingBoxResponse
from app.services.body_metrics import DerivedBodyMetrics


@dataclass(frozen=True, slots=True)
class ImageFeatureSet:
    """Body-fat-oriented visual features extracted from one image."""

    source_weight: float
    usable_for_body_fat_estimation: bool
    estimated_waist_to_hip_ratio: float | None
    estimated_shoulder_to_hip_ratio: float | None
    estimated_waist_to_bbox_height_ratio: float | None
    estimated_hip_to_bbox_height_ratio: float | None
    estimated_shoulder_to_bbox_height_ratio: float | None
    body_coverage_score: float
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class AggregatedVisualFeatures:
    """Aggregated visual feature set across all usable images."""

    usable_image_count: int
    total_image_count: int
    aggregate_quality_score: float | None
    feature_richness_score: float
    feature_consistency_score: float | None
    body_coverage_score: float | None
    estimated_waist_to_hip_ratio: float | None
    estimated_shoulder_to_hip_ratio: float | None
    estimated_waist_to_bbox_height_ratio: float | None
    estimated_hip_to_bbox_height_ratio: float | None
    estimated_shoulder_to_bbox_height_ratio: float | None
    warnings: list[str]


def build_image_feature_set(
    *,
    body_metrics: DerivedBodyMetrics,
    bbox: BoundingBoxResponse | None,
    quality_score: float | None,
    pose_confidence_score: float | None,
) -> ImageFeatureSet:
    """Create normalized visual ratios for a single image."""

    warnings = list(body_metrics.warnings)
    shoulder_to_hip_ratio = None
    if (
        body_metrics.estimated_shoulder_width_px is not None
        and body_metrics.estimated_hip_width_px is not None
        and body_metrics.estimated_hip_width_px > 0
    ):
        shoulder_to_hip_ratio = round(
            body_metrics.estimated_shoulder_width_px / body_metrics.estimated_hip_width_px,
            3,
        )

    waist_to_bbox_height_ratio = _safe_ratio(
        body_metrics.estimated_waist_width_px,
        bbox.height if bbox is not None else None,
    )
    hip_to_bbox_height_ratio = _safe_ratio(
        body_metrics.estimated_hip_width_px,
        bbox.height if bbox is not None else None,
    )
    shoulder_to_bbox_height_ratio = _safe_ratio(
        body_metrics.estimated_shoulder_width_px,
        bbox.height if bbox is not None else None,
    )
    body_coverage_score = _compute_body_coverage_score(body_metrics.lower_body_visibility_score)

    usable = (
        quality_score is not None
        and pose_confidence_score is not None
        and quality_score >= 0.35
        and pose_confidence_score >= 0.35
        and (
            body_metrics.estimated_waist_to_hip_ratio is not None
            or waist_to_bbox_height_ratio is not None
            or shoulder_to_hip_ratio is not None
        )
    )
    if body_coverage_score < 0.6:
        warnings.append("Body framing looks partial; confidence is reduced because lower-body coverage is weak.")
    if not usable:
        warnings.append("Image contributes weak or insufficient visual features for body fat estimation.")

    return ImageFeatureSet(
        source_weight=_compute_source_weight(quality_score, pose_confidence_score, body_coverage_score),
        usable_for_body_fat_estimation=usable,
        estimated_waist_to_hip_ratio=body_metrics.estimated_waist_to_hip_ratio,
        estimated_shoulder_to_hip_ratio=shoulder_to_hip_ratio,
        estimated_waist_to_bbox_height_ratio=waist_to_bbox_height_ratio,
        estimated_hip_to_bbox_height_ratio=hip_to_bbox_height_ratio,
        estimated_shoulder_to_bbox_height_ratio=shoulder_to_bbox_height_ratio,
        body_coverage_score=body_coverage_score,
        warnings=warnings,
    )


def aggregate_visual_features(feature_sets: list[ImageFeatureSet]) -> AggregatedVisualFeatures:
    """Aggregate body-fat-oriented visual features across multiple images."""

    usable_sets = [feature_set for feature_set in feature_sets if feature_set.usable_for_body_fat_estimation]
    warnings = _deduplicate_strings(
        warning
        for feature_set in feature_sets
        for warning in feature_set.warnings
    )

    if not usable_sets:
        warnings.append("No image provided reliable visual features for body fat estimation.")
        return AggregatedVisualFeatures(
            usable_image_count=0,
            total_image_count=len(feature_sets),
            aggregate_quality_score=None,
            feature_richness_score=0.0,
            feature_consistency_score=None,
            body_coverage_score=None,
            estimated_waist_to_hip_ratio=None,
            estimated_shoulder_to_hip_ratio=None,
            estimated_waist_to_bbox_height_ratio=None,
            estimated_hip_to_bbox_height_ratio=None,
            estimated_shoulder_to_bbox_height_ratio=None,
            warnings=_deduplicate_strings(warnings),
        )

    weights = [feature_set.source_weight for feature_set in usable_sets]
    aggregated_waist_to_hip_ratio = _weighted_mean(
        usable_sets,
        lambda item: item.estimated_waist_to_hip_ratio,
    )
    aggregated_shoulder_to_hip_ratio = _weighted_mean(
        usable_sets,
        lambda item: item.estimated_shoulder_to_hip_ratio,
    )
    aggregated_waist_to_bbox_height_ratio = _weighted_mean(
        usable_sets,
        lambda item: item.estimated_waist_to_bbox_height_ratio,
    )
    aggregated_hip_to_bbox_height_ratio = _weighted_mean(
        usable_sets,
        lambda item: item.estimated_hip_to_bbox_height_ratio,
    )
    aggregated_shoulder_to_bbox_height_ratio = _weighted_mean(
        usable_sets,
        lambda item: item.estimated_shoulder_to_bbox_height_ratio,
    )
    aggregated_body_coverage_score = _weighted_mean(
        usable_sets,
        lambda item: item.body_coverage_score,
    )

    return AggregatedVisualFeatures(
        usable_image_count=len(usable_sets),
        total_image_count=len(feature_sets),
        aggregate_quality_score=round(fmean(weights), 3),
        feature_richness_score=_compute_feature_richness(
            aggregated_waist_to_hip_ratio,
            aggregated_shoulder_to_hip_ratio,
            aggregated_waist_to_bbox_height_ratio,
            aggregated_hip_to_bbox_height_ratio,
            aggregated_shoulder_to_bbox_height_ratio,
        ),
        feature_consistency_score=_compute_feature_consistency(usable_sets),
        body_coverage_score=aggregated_body_coverage_score,
        estimated_waist_to_hip_ratio=aggregated_waist_to_hip_ratio,
        estimated_shoulder_to_hip_ratio=aggregated_shoulder_to_hip_ratio,
        estimated_waist_to_bbox_height_ratio=aggregated_waist_to_bbox_height_ratio,
        estimated_hip_to_bbox_height_ratio=aggregated_hip_to_bbox_height_ratio,
        estimated_shoulder_to_bbox_height_ratio=aggregated_shoulder_to_bbox_height_ratio,
        warnings=warnings,
    )


def _compute_source_weight(
    quality_score: float | None,
    pose_confidence_score: float | None,
    body_coverage_score: float,
) -> float:
    quality = quality_score if quality_score is not None else 0.0
    confidence = pose_confidence_score if pose_confidence_score is not None else 0.0
    return round(max(0.0, min(1.0, (quality * 0.45) + (confidence * 0.35) + (body_coverage_score * 0.20))), 3)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator, 3)


def _weighted_mean(
    feature_sets: list[ImageFeatureSet],
    accessor,
) -> float | None:
    pairs = [
        (accessor(feature_set), feature_set.source_weight)
        for feature_set in feature_sets
        if accessor(feature_set) is not None
    ]
    if not pairs:
        return None
    weighted_sum = sum(value * weight for value, weight in pairs)
    total_weight = sum(weight for _, weight in pairs)
    if total_weight == 0:
        return None
    return round(weighted_sum / total_weight, 3)


def _compute_feature_richness(*values: float | None) -> float:
    return round(sum(value is not None for value in values) / len(values), 3)


def _compute_feature_consistency(feature_sets: list[ImageFeatureSet]) -> float | None:
    if not feature_sets:
        return None
    if len(feature_sets) == 1:
        return 0.35

    feature_specs = [
        ("estimated_waist_to_hip_ratio", 0.10),
        ("estimated_shoulder_to_hip_ratio", 0.18),
        ("estimated_waist_to_bbox_height_ratio", 0.045),
        ("estimated_hip_to_bbox_height_ratio", 0.045),
        ("estimated_shoulder_to_bbox_height_ratio", 0.05),
    ]
    scores: list[float] = []
    for attribute_name, tolerance in feature_specs:
        values = [
            getattr(feature_set, attribute_name)
            for feature_set in feature_sets
            if getattr(feature_set, attribute_name) is not None
        ]
        if len(values) < 2:
            continue
        span = max(values) - min(values)
        scores.append(max(0.0, min(1.0, 1.0 - (span / tolerance))))

    if not scores:
        return 0.45
    return round(fmean(scores), 3)


def _deduplicate_strings(values) -> list[str]:
    return list(dict.fromkeys(values))


def _compute_body_coverage_score(lower_body_visibility_score: float | None) -> float:
    if lower_body_visibility_score is None:
        return 0.35
    return round(max(0.0, min(1.0, 0.35 + (lower_body_visibility_score * 0.65))), 3)
