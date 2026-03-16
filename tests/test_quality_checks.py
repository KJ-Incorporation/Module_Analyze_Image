"""Unit tests for quality-derived user feedback."""

from datetime import date

from app.services.body_metrics import BodyRegionStatus
from app.services.feature_engineering import AggregatedVisualFeatures
from app.services.quality_checks import build_analysis_feedback, build_coaching_feedback


def test_build_analysis_feedback_summarizes_reliable_and_weak_regions() -> None:
    feedback = build_analysis_feedback(
        has_successful_detection=True,
        overall_quality_score=0.81,
        confidence_score=0.78,
        estimated_body_fat_percent=15.4,
        analyzed_regions_summary=[
            BodyRegionStatus(
                key="torso",
                label="Torse",
                visible=True,
                confidence=0.94,
                taken_into_account=True,
            ),
            BodyRegionStatus(
                key="hips",
                label="Hanches",
                visible=True,
                confidence=0.92,
                taken_into_account=True,
            ),
            BodyRegionStatus(
                key="left_lower_leg",
                label="Mollet / jambe gauche",
                visible=False,
                confidence=0.22,
                taken_into_account=False,
            ),
        ],
    )

    assert "Good analysis base." in feedback
    assert "Reliable zones: Torse and Hanches." in feedback
    assert "Weak or missing zones: Mollet / jambe gauche." in feedback


def test_build_analysis_feedback_handles_missing_detection() -> None:
    feedback = build_analysis_feedback(
        has_successful_detection=False,
        overall_quality_score=None,
        confidence_score=None,
        estimated_body_fat_percent=None,
        analyzed_regions_summary=[],
    )

    assert feedback == "No reliable body was detected. Retake at least one clear, centered photo."


def test_build_coaching_feedback_returns_projection_when_inputs_are_strong() -> None:
    feedback = build_coaching_feedback(
        sex="male",
        confidence_score=0.82,
        estimated_body_fat_percent=16.8,
        estimated_lean_mass_kg=66.0,
        weight_kg=79.5,
        visual_features=AggregatedVisualFeatures(
            usable_image_count=3,
            total_image_count=3,
            aggregate_quality_score=0.86,
            feature_richness_score=1.0,
            feature_consistency_score=0.84,
            body_coverage_score=0.92,
            estimated_waist_to_hip_ratio=0.93,
            estimated_shoulder_to_hip_ratio=1.17,
            estimated_waist_to_bbox_height_ratio=0.208,
            estimated_hip_to_bbox_height_ratio=0.22,
            estimated_shoulder_to_bbox_height_ratio=0.245,
            warnings=[],
        ),
        reference_date=date(2026, 3, 9),
    )

    assert "Mild abdominal fat retention is still likely." in feedback
    assert "0.5 kg per week" in feedback
    assert "April 2026" in feedback


def test_build_coaching_feedback_rejects_projection_when_confidence_is_too_low() -> None:
    feedback = build_coaching_feedback(
        sex="male",
        confidence_score=0.41,
        estimated_body_fat_percent=16.8,
        estimated_lean_mass_kg=66.0,
        weight_kg=79.5,
        visual_features=AggregatedVisualFeatures(
            usable_image_count=1,
            total_image_count=1,
            aggregate_quality_score=0.54,
            feature_richness_score=0.6,
            feature_consistency_score=0.3,
            body_coverage_score=0.42,
            estimated_waist_to_hip_ratio=0.93,
            estimated_shoulder_to_hip_ratio=1.17,
            estimated_waist_to_bbox_height_ratio=0.208,
            estimated_hip_to_bbox_height_ratio=0.22,
            estimated_shoulder_to_bbox_height_ratio=0.245,
            warnings=[],
        ),
        reference_date=date(2026, 3, 9),
    )

    assert "too low-confidence" in feedback
