"""Unit tests for visual feature engineering."""

from app.schemas.response import BoundingBoxResponse
from app.services.body_metrics import DerivedBodyMetrics
from app.services.feature_engineering import aggregate_visual_features, build_image_feature_set


def test_build_image_feature_set_marks_image_usable_when_signals_are_present() -> None:
    body_metrics = DerivedBodyMetrics(
        estimated_shoulder_width_px=120.0,
        estimated_hip_width_px=100.0,
        estimated_waist_width_px=95.0,
        estimated_waist_to_hip_ratio=0.95,
        posture_summary="upright / balanced",
        lower_body_visibility_score=0.92,
        warnings=[],
    )
    bbox = BoundingBoxResponse(x_min=10, y_min=20, x_max=210, y_max=420, width=200, height=400)

    feature_set = build_image_feature_set(
        body_metrics=body_metrics,
        bbox=bbox,
        quality_score=0.82,
        pose_confidence_score=0.88,
    )

    assert feature_set.usable_for_body_fat_estimation is True
    assert feature_set.estimated_waist_to_hip_ratio == 0.95
    assert feature_set.estimated_shoulder_to_hip_ratio == 1.2
    assert feature_set.estimated_waist_to_bbox_height_ratio == 0.237
    assert feature_set.body_coverage_score > 0.9


def test_aggregate_visual_features_computes_weighted_means() -> None:
    first = build_image_feature_set(
        body_metrics=DerivedBodyMetrics(120.0, 100.0, 95.0, 0.95, "upright / balanced", 0.92, []),
        bbox=BoundingBoxResponse(x_min=0, y_min=0, x_max=200, y_max=400, width=200, height=400),
        quality_score=0.8,
        pose_confidence_score=0.9,
    )
    second = build_image_feature_set(
        body_metrics=DerivedBodyMetrics(118.0, 102.0, 92.0, 0.902, "upright / balanced", 0.88, []),
        bbox=BoundingBoxResponse(x_min=0, y_min=0, x_max=220, y_max=440, width=220, height=440),
        quality_score=0.7,
        pose_confidence_score=0.8,
    )

    aggregated = aggregate_visual_features([first, second])

    assert aggregated.usable_image_count == 2
    assert aggregated.aggregate_quality_score is not None
    assert aggregated.feature_richness_score == 1.0
    assert aggregated.feature_consistency_score is not None
    assert aggregated.body_coverage_score is not None
    assert aggregated.body_coverage_score > 0.85
    assert 0.92 < aggregated.estimated_waist_to_hip_ratio < 0.94
    assert aggregated.estimated_shoulder_to_hip_ratio is not None


def test_aggregate_visual_features_returns_nulls_when_no_usable_images() -> None:
    weak_feature_set = build_image_feature_set(
        body_metrics=DerivedBodyMetrics(None, None, None, None, None, None, ["Torso not visible"]),
        bbox=None,
        quality_score=0.2,
        pose_confidence_score=0.2,
    )

    aggregated = aggregate_visual_features([weak_feature_set])

    assert aggregated.usable_image_count == 0
    assert aggregated.feature_richness_score == 0.0
    assert aggregated.body_coverage_score is None
    assert aggregated.estimated_waist_to_hip_ratio is None
    assert "No image provided reliable visual features for body fat estimation." in aggregated.warnings


def test_build_image_feature_set_penalizes_partial_body_coverage() -> None:
    partial_body_metrics = DerivedBodyMetrics(
        estimated_shoulder_width_px=120.0,
        estimated_hip_width_px=100.0,
        estimated_waist_width_px=95.0,
        estimated_waist_to_hip_ratio=0.95,
        posture_summary="upright / balanced",
        lower_body_visibility_score=None,
        warnings=[],
    )

    feature_set = build_image_feature_set(
        body_metrics=partial_body_metrics,
        bbox=BoundingBoxResponse(x_min=10, y_min=20, x_max=210, y_max=420, width=200, height=400),
        quality_score=0.82,
        pose_confidence_score=0.88,
    )

    assert feature_set.body_coverage_score == 0.35
    assert any("partial" in warning.lower() for warning in feature_set.warnings)


def test_aggregate_visual_features_tracks_view_diversity_and_pose_quality() -> None:
    front = build_image_feature_set(
        body_metrics=DerivedBodyMetrics(124.0, 104.0, 92.0, 0.885, "upright / balanced", 0.92, []),
        bbox=BoundingBoxResponse(x_min=0, y_min=0, x_max=220, y_max=480, width=220, height=480),
        quality_score=0.86,
        pose_confidence_score=0.9,
        posture_summary="upright / balanced",
    )
    side = build_image_feature_set(
        body_metrics=DerivedBodyMetrics(82.0, 74.0, 70.0, 0.946, "slight asymmetry", 0.88, []),
        bbox=BoundingBoxResponse(x_min=0, y_min=0, x_max=150, y_max=470, width=150, height=470),
        quality_score=0.81,
        pose_confidence_score=0.86,
        posture_summary="slight asymmetry",
    )

    aggregated = aggregate_visual_features([front, side])

    assert aggregated.front_view_count == 1
    assert aggregated.side_view_count == 1
    assert aggregated.view_diversity_score == 1.0
    assert aggregated.pose_neutrality_score is not None
    assert aggregated.pose_neutrality_score > 0.8
