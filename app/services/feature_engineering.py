"""Feature engineering for multi-image body fat inference."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    estimated_view_type: str | None = None
    pose_neutrality_score: float | None = None
    warnings: list[str] = field(default_factory=list)


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
    taper_score: float | None = None
    central_fat_score: float | None = None
    definition_score: float | None = None
    frame_score: float | None = None
    reliability_score: float | None = None
    front_view_count: int = 0
    side_view_count: int = 0
    three_quarter_view_count: int = 0
    view_diversity_score: float | None = None
    pose_neutrality_score: float | None = None
    warnings: list[str] = field(default_factory=list)


def build_image_feature_set(
    *,
    body_metrics: DerivedBodyMetrics,
    bbox: BoundingBoxResponse | None,
    quality_score: float | None,
    pose_confidence_score: float | None,
    posture_summary: str | None = None,
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
    estimated_view_type = _estimate_view_type(
        shoulder_to_hip_ratio=shoulder_to_hip_ratio,
        shoulder_to_bbox_height_ratio=shoulder_to_bbox_height_ratio,
        hip_to_bbox_height_ratio=hip_to_bbox_height_ratio,
    )
    pose_neutrality_score = _estimate_pose_neutrality_score(posture_summary)

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
    if estimated_view_type == "unknown":
        warnings.append("Body angle is hard to classify; multi-view confidence is reduced.")
    if (pose_neutrality_score or 0.0) < 0.5:
        warnings.append("Pose looks less neutral; body-shape interpretation may be less stable.")
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
        estimated_view_type=estimated_view_type,
        pose_neutrality_score=pose_neutrality_score,
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
            taper_score=None,
            central_fat_score=None,
            definition_score=None,
            frame_score=None,
            reliability_score=0.0,
            front_view_count=0,
            side_view_count=0,
            three_quarter_view_count=0,
            view_diversity_score=0.0,
            pose_neutrality_score=0.0,
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
    aggregated_pose_neutrality_score = _weighted_mean(
        usable_sets,
        lambda item: item.pose_neutrality_score,
    )
    front_view_count = sum(1 for item in usable_sets if item.estimated_view_type == "front")
    side_view_count = sum(1 for item in usable_sets if item.estimated_view_type == "side")
    three_quarter_view_count = sum(1 for item in usable_sets if item.estimated_view_type == "three_quarter")
    view_diversity_score = _compute_view_diversity_score(
        usable_image_count=len(usable_sets),
        front_view_count=front_view_count,
        side_view_count=side_view_count,
        three_quarter_view_count=three_quarter_view_count,
    )

    taper_score = _compute_taper_score(
        aggregated_waist_to_hip_ratio,
        aggregated_shoulder_to_hip_ratio,
    )
    central_fat_score = _compute_central_fat_score(
        aggregated_waist_to_hip_ratio,
        aggregated_waist_to_bbox_height_ratio,
    )
    frame_score = _compute_frame_score(
        aggregated_shoulder_to_hip_ratio,
        aggregated_shoulder_to_bbox_height_ratio,
    )
    reliability_score = _compute_reliability_score(
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
        view_diversity_score=view_diversity_score,
        pose_neutrality_score=aggregated_pose_neutrality_score,
    )
    definition_score = _compute_definition_score(
        taper_score=taper_score,
        central_fat_score=central_fat_score,
        body_coverage_score=aggregated_body_coverage_score,
        feature_consistency_score=_compute_feature_consistency(usable_sets),
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
        taper_score=taper_score,
        central_fat_score=central_fat_score,
        definition_score=definition_score,
        frame_score=frame_score,
        reliability_score=reliability_score,
        front_view_count=front_view_count,
        side_view_count=side_view_count,
        three_quarter_view_count=three_quarter_view_count,
        view_diversity_score=view_diversity_score,
        pose_neutrality_score=aggregated_pose_neutrality_score,
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


def _compute_taper_score(
    waist_to_hip_ratio: float | None,
    shoulder_to_hip_ratio: float | None,
) -> float | None:
    components: list[float] = []
    if waist_to_hip_ratio is not None:
        components.append(_inverse_normalized_value(waist_to_hip_ratio, 0.75, 1.0))
    if shoulder_to_hip_ratio is not None:
        components.append(_normalized_value(shoulder_to_hip_ratio, 0.95, 1.28))
    if not components:
        return None
    return round(fmean(components), 3)


def _compute_central_fat_score(
    waist_to_hip_ratio: float | None,
    waist_to_bbox_height_ratio: float | None,
) -> float | None:
    components: list[float] = []
    if waist_to_hip_ratio is not None:
        components.append(_normalized_value(waist_to_hip_ratio, 0.78, 0.98))
    if waist_to_bbox_height_ratio is not None:
        components.append(_normalized_value(waist_to_bbox_height_ratio, 0.16, 0.27))
    if not components:
        return None
    return round(fmean(components), 3)


def _compute_definition_score(
    *,
    taper_score: float | None,
    central_fat_score: float | None,
    body_coverage_score: float | None,
    feature_consistency_score: float | None,
) -> float | None:
    components: list[float] = []
    if taper_score is not None:
        components.append(taper_score)
    if central_fat_score is not None:
        components.append(1.0 - central_fat_score)
    if body_coverage_score is not None:
        components.append(body_coverage_score)
    if feature_consistency_score is not None:
        components.append(feature_consistency_score)
    if not components:
        return None
    return round(fmean(components), 3)


def _compute_frame_score(
    shoulder_to_hip_ratio: float | None,
    shoulder_to_bbox_height_ratio: float | None,
) -> float | None:
    components: list[float] = []
    if shoulder_to_hip_ratio is not None:
        components.append(_normalized_value(shoulder_to_hip_ratio, 0.96, 1.25))
    if shoulder_to_bbox_height_ratio is not None:
        components.append(_normalized_value(shoulder_to_bbox_height_ratio, 0.19, 0.29))
    if not components:
        return None
    return round(fmean(components), 3)


def _compute_reliability_score(
    *,
    usable_image_count: int,
    total_image_count: int,
    aggregate_quality_score: float | None,
    feature_richness_score: float,
    feature_consistency_score: float | None,
    body_coverage_score: float | None,
    view_diversity_score: float | None,
    pose_neutrality_score: float | None,
) -> float:
    image_support = 0.0 if total_image_count == 0 else min(1.0, usable_image_count / max(1, min(3, total_image_count)))
    return round(
        max(
            0.0,
            min(
                1.0,
                ((aggregate_quality_score or 0.0) * 0.24)
                + (feature_richness_score * 0.18)
                + ((feature_consistency_score or 0.0) * 0.17)
                + ((body_coverage_score or 0.0) * 0.20)
                + ((view_diversity_score or 0.0) * 0.11)
                + ((pose_neutrality_score or 0.0) * 0.10)
                + (image_support * 0.10),
            ),
        ),
        3,
    )


def _deduplicate_strings(values) -> list[str]:
    return list(dict.fromkeys(values))


def _compute_body_coverage_score(lower_body_visibility_score: float | None) -> float:
    if lower_body_visibility_score is None:
        return 0.35
    return round(max(0.0, min(1.0, 0.35 + (lower_body_visibility_score * 0.65))), 3)


def _estimate_view_type(
    *,
    shoulder_to_hip_ratio: float | None,
    shoulder_to_bbox_height_ratio: float | None,
    hip_to_bbox_height_ratio: float | None,
) -> str:
    if shoulder_to_bbox_height_ratio is None or hip_to_bbox_height_ratio is None:
        return "unknown"
    if shoulder_to_bbox_height_ratio >= 0.225 and hip_to_bbox_height_ratio >= 0.19:
        return "front"
    if shoulder_to_bbox_height_ratio <= 0.19 and hip_to_bbox_height_ratio <= 0.18:
        return "side"
    if shoulder_to_hip_ratio is not None and shoulder_to_hip_ratio >= 1.02:
        return "three_quarter"
    return "unknown"


def _estimate_pose_neutrality_score(posture_summary: str | None) -> float:
    if posture_summary == "upright / balanced":
        return 1.0
    if posture_summary == "slight asymmetry":
        return 0.72
    if posture_summary == "possible tilt / asymmetry":
        return 0.45
    if posture_summary == "possible lateral lean":
        return 0.35
    return 0.4


def _compute_view_diversity_score(
    *,
    usable_image_count: int,
    front_view_count: int,
    side_view_count: int,
    three_quarter_view_count: int,
) -> float:
    if usable_image_count == 0:
        return 0.0
    if front_view_count >= 1 and side_view_count >= 1:
        return 1.0
    if (front_view_count >= 1 and three_quarter_view_count >= 1) or (side_view_count >= 1 and three_quarter_view_count >= 1):
        return 0.82
    if usable_image_count >= 2:
        return 0.58
    if front_view_count == 1:
        return 0.48
    if side_view_count == 1:
        return 0.44
    if three_quarter_view_count == 1:
        return 0.46
    return 0.35


def _normalized_value(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _inverse_normalized_value(value: float, low: float, high: float) -> float:
    return 1.0 - _normalized_value(value, low, high)
