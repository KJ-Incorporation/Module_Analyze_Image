"""Unit tests for quality-derived user feedback."""

from datetime import date

from app.services.body_metrics import BodyRegionStatus
from app.services.feature_engineering import AggregatedVisualFeatures
from app.services.quality_checks import (
    build_analysis_blocks,
    build_analysis_feedback,
    build_analysis_notes,
    build_coaching_feedback,
    build_scan_profile,
)


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
        sex="male",
        visual_features=AggregatedVisualFeatures(
            usable_image_count=2,
            total_image_count=2,
            aggregate_quality_score=0.82,
            feature_richness_score=1.0,
            feature_consistency_score=0.79,
            body_coverage_score=0.91,
            estimated_waist_to_hip_ratio=0.92,
            estimated_shoulder_to_hip_ratio=1.18,
            estimated_waist_to_bbox_height_ratio=0.205,
            estimated_hip_to_bbox_height_ratio=0.22,
            estimated_shoulder_to_bbox_height_ratio=0.25,
            warnings=[],
        ),
    )

    assert "La base d'analyse est vraiment solide." in feedback or "L'analyse est exploitable" in feedback
    assert "Zones les plus fiables : Torse et Hanches." in feedback
    assert "Zones encore faibles ou partielles : Mollet / jambe gauche." in feedback
    assert "taille ou de l'abdomen" in feedback or "residuel" in feedback


def test_build_analysis_feedback_handles_missing_detection() -> None:
    feedback = build_analysis_feedback(
        has_successful_detection=False,
        overall_quality_score=None,
        confidence_score=None,
        estimated_body_fat_percent=None,
        analyzed_regions_summary=[],
    )

    assert feedback == "Aucun corps n'a ete detecte de maniere suffisamment fiable. Reprends au moins une photo nette et bien cadree."


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

    assert "taille" in feedback or "abdomen" in feedback
    assert "0.5 kg par semaine" in feedback
    assert "2026" in feedback


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

    assert "trop faible" in feedback


def test_build_analysis_notes_returns_three_product_sections() -> None:
    notes = build_analysis_notes(
        has_successful_detection=True,
        overall_quality_score=0.89,
        confidence_score=0.78,
        estimated_body_fat_percent=15.4,
        estimated_lean_mass_kg=63.45,
        weight_kg=75.0,
        sex="male",
        analyzed_regions_summary=[
            BodyRegionStatus(
                key="torso",
                label="Torse",
                visible=True,
                confidence=0.98,
                taken_into_account=True,
            ),
            BodyRegionStatus(
                key="waist",
                label="Taille",
                visible=True,
                confidence=0.97,
                taken_into_account=True,
            ),
        ],
        visual_features=AggregatedVisualFeatures(
            usable_image_count=2,
            total_image_count=2,
            aggregate_quality_score=0.89,
            feature_richness_score=1.0,
            feature_consistency_score=0.75,
            body_coverage_score=0.92,
            estimated_waist_to_hip_ratio=0.93,
            estimated_shoulder_to_hip_ratio=1.1,
            estimated_waist_to_bbox_height_ratio=0.21,
            estimated_hip_to_bbox_height_ratio=0.22,
            estimated_shoulder_to_bbox_height_ratio=0.24,
            front_view_count=1,
            side_view_count=1,
            three_quarter_view_count=0,
            view_diversity_score=1.0,
            pose_neutrality_score=0.92,
            warnings=[],
        ),
        reference_date=date(2026, 3, 16),
    )

    assert notes.attention.title in {
        "Graisse abdominale encore visible",
        "Taille encore chargee",
        "Body fat encore trop present",
        "Definition encore trop faible",
    }
    assert notes.parfait.title in {"Lecture fiable", "Base athletique credible", "Points forts identifies", "Structure solide"}
    assert notes.progression.title in {"Objectif definition", "Prochain levier concret", "Levier principal"}
    assert "taille" in notes.attention.message or "abdominale" in notes.attention.message or "gras" in notes.attention.message
    assert "base" in notes.parfait.message.lower() or "scan" in notes.parfait.message.lower() or "zones utiles" in notes.parfait.message.lower()
    assert "0.5 kg par semaine" in notes.progression.message
    assert "2026" in notes.progression.message
    assert "confidence " not in notes.attention.message.lower()
    assert "confidence " not in notes.parfait.message.lower()
    assert "confidence " not in notes.progression.message.lower()


def test_build_scan_profile_returns_premium_structured_summary() -> None:
    profile = build_scan_profile(
        has_successful_detection=True,
        confidence_score=0.84,
        overall_quality_score=0.87,
        estimated_body_fat_percent=15.2,
        analyzed_regions_summary=[
            BodyRegionStatus(
                key="torso",
                label="Torse",
                visible=True,
                confidence=0.98,
                taken_into_account=True,
            ),
            BodyRegionStatus(
                key="waist",
                label="Taille",
                visible=True,
                confidence=0.97,
                taken_into_account=True,
            ),
        ],
        sex="male",
        visual_features=AggregatedVisualFeatures(
            usable_image_count=3,
            total_image_count=3,
            aggregate_quality_score=0.86,
            feature_richness_score=1.0,
            feature_consistency_score=0.82,
            body_coverage_score=0.93,
            estimated_waist_to_hip_ratio=0.9,
            estimated_shoulder_to_hip_ratio=1.16,
            estimated_waist_to_bbox_height_ratio=0.198,
            estimated_hip_to_bbox_height_ratio=0.22,
            estimated_shoulder_to_bbox_height_ratio=0.255,
            taper_score=0.73,
            central_fat_score=0.46,
            definition_score=0.74,
            frame_score=0.79,
            reliability_score=0.86,
            front_view_count=2,
            side_view_count=1,
            three_quarter_view_count=0,
            view_diversity_score=1.0,
            pose_neutrality_score=0.9,
            warnings=[],
        ),
    )

    assert profile.reliability_level == "high"
    assert profile.confidence_label in {"Tres fiable", "Fiable"}
    assert "scan" in profile.confidence_message.lower()
    assert profile.definition_level in {"moderate_to_good", "very_good"}
    assert profile.frame_assessment in {"balanced_frame", "strong_frame"}
    assert profile.view_coverage in {"front_and_side_available", "multi_angle_supported"}
    assert profile.pose_quality in {"neutral", "mostly_neutral"}
    assert profile.scan_readiness == "ready_for_actionable_feedback"
    assert profile.best_next_focus in {"continue_improving_definition", "maintain_and_refine"}
    assert profile.dominant_strength in {"scan_clarity", "visible_definition", "upper_body_structure", "multi_view_support"}
    assert profile.dominant_limitation in {"central_fat", "minor_residual_gap", "lack_of_definition"}
    assert "scan" in profile.summary.lower()


def test_build_analysis_blocks_returns_expanded_frontend_cards() -> None:
    notes = build_analysis_notes(
        has_successful_detection=True,
        overall_quality_score=0.88,
        confidence_score=0.81,
        estimated_body_fat_percent=15.1,
        estimated_lean_mass_kg=64.0,
        weight_kg=75.0,
        sex="male",
        analyzed_regions_summary=[
            BodyRegionStatus(
                key="torso",
                label="Torse",
                visible=True,
                confidence=0.97,
                taken_into_account=True,
            ),
            BodyRegionStatus(
                key="waist",
                label="Taille",
                visible=True,
                confidence=0.96,
                taken_into_account=True,
            ),
        ],
        visual_features=AggregatedVisualFeatures(
            usable_image_count=2,
            total_image_count=2,
            aggregate_quality_score=0.88,
            feature_richness_score=1.0,
            feature_consistency_score=0.8,
            body_coverage_score=0.93,
            estimated_waist_to_hip_ratio=0.91,
            estimated_shoulder_to_hip_ratio=1.15,
            estimated_waist_to_bbox_height_ratio=0.198,
            estimated_hip_to_bbox_height_ratio=0.22,
            estimated_shoulder_to_bbox_height_ratio=0.25,
            front_view_count=1,
            side_view_count=1,
            view_diversity_score=1.0,
            pose_neutrality_score=0.88,
            warnings=[],
        ),
        reference_date=date(2026, 3, 16),
    )
    profile = build_scan_profile(
        has_successful_detection=True,
        confidence_score=0.81,
        overall_quality_score=0.88,
        estimated_body_fat_percent=15.1,
        analyzed_regions_summary=[
            BodyRegionStatus(
                key="torso",
                label="Torse",
                visible=True,
                confidence=0.97,
                taken_into_account=True,
            ),
            BodyRegionStatus(
                key="waist",
                label="Taille",
                visible=True,
                confidence=0.96,
                taken_into_account=True,
            ),
        ],
        sex="male",
        visual_features=AggregatedVisualFeatures(
            usable_image_count=2,
            total_image_count=2,
            aggregate_quality_score=0.88,
            feature_richness_score=1.0,
            feature_consistency_score=0.8,
            body_coverage_score=0.93,
            estimated_waist_to_hip_ratio=0.91,
            estimated_shoulder_to_hip_ratio=1.15,
            estimated_waist_to_bbox_height_ratio=0.198,
            estimated_hip_to_bbox_height_ratio=0.22,
            estimated_shoulder_to_bbox_height_ratio=0.25,
            taper_score=0.74,
            central_fat_score=0.47,
            definition_score=0.73,
            frame_score=0.77,
            reliability_score=0.84,
            front_view_count=1,
            side_view_count=1,
            view_diversity_score=1.0,
            pose_neutrality_score=0.88,
            warnings=[],
        ),
    )

    blocks = build_analysis_blocks(
        analysis_notes=notes,
        scan_profile=profile,
        confidence_score=0.81,
        overall_quality_score=0.88,
        estimated_body_fat_percent=15.1,
        sex="male",
    )

    assert blocks.overview.title == "Lecture globale"
    assert blocks.truth.title
    assert blocks.strength.title
    assert blocks.limitation.title
    assert blocks.next_focus.title
    assert blocks.scan_quality.title == "Niveau de fiabilite"
    assert "scan" in blocks.overview.message.lower()
    assert "confidence=" not in blocks.scan_quality.message.lower()
    assert "quality=" not in blocks.scan_quality.message.lower()


def test_build_analysis_notes_explicitly_handles_upper_body_only_scan() -> None:
    notes = build_analysis_notes(
        has_successful_detection=True,
        overall_quality_score=0.82,
        confidence_score=0.71,
        estimated_body_fat_percent=16.2,
        estimated_lean_mass_kg=63.0,
        weight_kg=75.0,
        sex="male",
        analyzed_regions_summary=[
            BodyRegionStatus(key="head", label="Tete", visible=True, confidence=0.99, taken_into_account=True),
            BodyRegionStatus(key="torso", label="Torse", visible=True, confidence=0.98, taken_into_account=True),
            BodyRegionStatus(key="waist", label="Taille", visible=True, confidence=0.97, taken_into_account=True),
            BodyRegionStatus(key="hips", label="Hanches", visible=True, confidence=0.92, taken_into_account=True),
            BodyRegionStatus(key="left_thigh", label="Cuisse gauche", visible=False, confidence=0.22, taken_into_account=False),
            BodyRegionStatus(key="right_thigh", label="Cuisse droite", visible=False, confidence=0.18, taken_into_account=False),
            BodyRegionStatus(key="left_lower_leg", label="Mollet / jambe gauche", visible=False, confidence=0.09, taken_into_account=False),
            BodyRegionStatus(key="right_lower_leg", label="Mollet / jambe droite", visible=False, confidence=0.08, taken_into_account=False),
        ],
        visual_features=AggregatedVisualFeatures(
            usable_image_count=1,
            total_image_count=1,
            aggregate_quality_score=0.82,
            feature_richness_score=0.9,
            feature_consistency_score=0.35,
            body_coverage_score=0.41,
            estimated_waist_to_hip_ratio=0.94,
            estimated_shoulder_to_hip_ratio=1.11,
            estimated_waist_to_bbox_height_ratio=0.21,
            estimated_hip_to_bbox_height_ratio=0.22,
            estimated_shoulder_to_bbox_height_ratio=0.24,
            view_diversity_score=0.48,
            pose_neutrality_score=0.88,
            warnings=[],
        ),
        reference_date=date(2026, 3, 16),
    )

    assert notes.attention.title == "Bas du corps encore absent"
    assert notes.parfait.title == "Haut du corps bien capte"
    assert notes.progression.title == "Montrer le bas du corps"
    assert "haut du corps" in notes.attention.message.lower()
    assert "haut du corps" in notes.parfait.message.lower()
    assert "hanches, cuisses et jambes" in notes.progression.message.lower()
    assert "confidence " not in notes.attention.message.lower()
