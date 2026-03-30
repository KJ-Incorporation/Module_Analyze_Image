"""Unit tests for top-level body fat and somatotype estimation."""

from app.services.bodyfat_estimator import estimate_body_fat, estimate_somatotype
from app.services.feature_engineering import AggregatedVisualFeatures


def test_estimate_body_fat_returns_estimate_when_inputs_are_sufficient() -> None:
    visual_features = AggregatedVisualFeatures(
        usable_image_count=2,
        total_image_count=2,
        aggregate_quality_score=0.82,
        feature_richness_score=1.0,
        feature_consistency_score=0.84,
        body_coverage_score=0.93,
        estimated_waist_to_hip_ratio=0.88,
        estimated_shoulder_to_hip_ratio=1.08,
        estimated_waist_to_bbox_height_ratio=0.2,
        estimated_hip_to_bbox_height_ratio=0.22,
        estimated_shoulder_to_bbox_height_ratio=0.24,
        warnings=[],
    )

    result = estimate_body_fat(
        age=31,
        sex="female",
        height_cm=168.0,
        weight_kg=63.5,
        visual_features=visual_features,
        model_version="heuristic-bodyfat-v0.2.0",
    )

    assert result.bmi == 22.5
    assert result.estimated_body_fat_percent is not None
    assert result.estimated_fat_mass_kg is not None
    assert result.estimated_lean_mass_kg is not None
    assert result.confidence_score is not None
    assert result.model_version == "heuristic-bodyfat-v0.2.0"
    assert 18.0 <= result.estimated_body_fat_percent <= 26.0


def test_estimate_body_fat_returns_null_when_required_inputs_are_missing() -> None:
    visual_features = AggregatedVisualFeatures(
        usable_image_count=1,
        total_image_count=1,
        aggregate_quality_score=0.7,
        feature_richness_score=1.0,
        feature_consistency_score=0.35,
        body_coverage_score=0.52,
        estimated_waist_to_hip_ratio=0.9,
        estimated_shoulder_to_hip_ratio=1.1,
        estimated_waist_to_bbox_height_ratio=0.19,
        estimated_hip_to_bbox_height_ratio=0.21,
        estimated_shoulder_to_bbox_height_ratio=0.23,
        warnings=[],
    )

    result = estimate_body_fat(
        age=None,
        sex="female",
        height_cm=168.0,
        weight_kg=63.5,
        visual_features=visual_features,
        model_version="heuristic-bodyfat-v0.2.0",
    )

    assert result.bmi == 22.5
    assert result.estimated_body_fat_percent is None
    assert result.estimated_fat_mass_kg is None
    assert result.estimated_lean_mass_kg is None
    assert "Age is required for body fat estimation." in result.warnings


def test_estimate_body_fat_returns_low_confidence_when_visual_support_is_missing() -> None:
    visual_features = AggregatedVisualFeatures(
        usable_image_count=0,
        total_image_count=2,
        aggregate_quality_score=None,
        feature_richness_score=0.0,
        feature_consistency_score=None,
        body_coverage_score=None,
        estimated_waist_to_hip_ratio=None,
        estimated_shoulder_to_hip_ratio=None,
        estimated_waist_to_bbox_height_ratio=None,
        estimated_hip_to_bbox_height_ratio=None,
        estimated_shoulder_to_bbox_height_ratio=None,
        warnings=["No image provided reliable visual features for body fat estimation."],
    )

    result = estimate_body_fat(
        age=40,
        sex="male",
        height_cm=180.0,
        weight_kg=82.0,
        visual_features=visual_features,
        model_version="heuristic-bodyfat-v0.2.0",
    )

    assert result.bmi == 25.31
    assert result.estimated_body_fat_percent is None
    assert result.confidence_score is not None
    assert result.confidence_score <= 0.35


def test_estimate_body_fat_can_stay_in_lean_range_for_strong_male_visual_features() -> None:
    visual_features = AggregatedVisualFeatures(
        usable_image_count=3,
        total_image_count=3,
        aggregate_quality_score=0.88,
        feature_richness_score=1.0,
        feature_consistency_score=0.91,
        body_coverage_score=0.94,
        estimated_waist_to_hip_ratio=0.83,
        estimated_shoulder_to_hip_ratio=1.24,
        estimated_waist_to_bbox_height_ratio=0.171,
        estimated_hip_to_bbox_height_ratio=0.21,
        estimated_shoulder_to_bbox_height_ratio=0.262,
        warnings=[],
    )

    result = estimate_body_fat(
        age=29,
        sex="male",
        height_cm=178.0,
        weight_kg=73.0,
        visual_features=visual_features,
        model_version="heuristic-bodyfat-v0.3.0",
    )

    assert result.bmi == 23.04
    assert result.estimated_body_fat_percent is not None
    assert 10.0 <= result.estimated_body_fat_percent <= 15.5
    assert result.confidence_score is not None
    assert result.confidence_score >= 0.8


def test_estimate_body_fat_reduces_confidence_when_body_coverage_is_partial() -> None:
    visual_features = AggregatedVisualFeatures(
        usable_image_count=1,
        total_image_count=1,
        aggregate_quality_score=0.84,
        feature_richness_score=1.0,
        feature_consistency_score=0.35,
        body_coverage_score=0.35,
        estimated_waist_to_hip_ratio=0.84,
        estimated_shoulder_to_hip_ratio=1.22,
        estimated_waist_to_bbox_height_ratio=0.176,
        estimated_hip_to_bbox_height_ratio=0.208,
        estimated_shoulder_to_bbox_height_ratio=0.26,
        warnings=["Body framing looks partial; confidence is reduced because lower-body coverage is weak."],
    )

    result = estimate_body_fat(
        age=29,
        sex="male",
        height_cm=178.0,
        weight_kg=73.0,
        visual_features=visual_features,
        model_version="heuristic-bodyfat-v0.3.0",
    )

    assert result.estimated_body_fat_percent is not None
    assert result.confidence_score is not None
    assert result.confidence_score < 0.7


def test_estimate_somatotype_returns_mesomorphic_bias_for_athletic_male_profile() -> None:
    visual_features = AggregatedVisualFeatures(
        usable_image_count=2,
        total_image_count=2,
        aggregate_quality_score=0.86,
        feature_richness_score=1.0,
        feature_consistency_score=0.83,
        body_coverage_score=0.92,
        estimated_waist_to_hip_ratio=0.85,
        estimated_shoulder_to_hip_ratio=1.22,
        estimated_waist_to_bbox_height_ratio=0.18,
        estimated_hip_to_bbox_height_ratio=0.22,
        estimated_shoulder_to_bbox_height_ratio=0.26,
        warnings=[],
    )

    somatotype = estimate_somatotype(
        sex="male",
        bmi=23.1,
        visual_features=visual_features,
        body_fat_percent=13.8,
    )

    assert somatotype.primary == "mesomorph"
    assert somatotype.secondary in {"ectomorph", "endomorph"}
    assert somatotype.confidence is not None
    assert somatotype.mesomorph_score > somatotype.endomorph_score
    assert somatotype.notes.startswith("Estime a partir")


def test_estimate_somatotype_returns_unavailable_when_visual_support_is_missing() -> None:
    visual_features = AggregatedVisualFeatures(
        usable_image_count=0,
        total_image_count=1,
        aggregate_quality_score=None,
        feature_richness_score=0.0,
        feature_consistency_score=None,
        body_coverage_score=None,
        estimated_waist_to_hip_ratio=None,
        estimated_shoulder_to_hip_ratio=None,
        estimated_waist_to_bbox_height_ratio=None,
        estimated_hip_to_bbox_height_ratio=None,
        estimated_shoulder_to_bbox_height_ratio=None,
        warnings=[],
    )

    somatotype = estimate_somatotype(
        sex="male",
        bmi=None,
        visual_features=visual_features,
        body_fat_percent=None,
    )

    assert somatotype.primary is None
    assert somatotype.confidence == 0.0
    assert "pas encore assez fiables" in somatotype.notes
