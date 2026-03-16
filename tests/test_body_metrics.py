"""Unit tests for pure body metric calculations."""

from app.services.body_metrics import (
    LANDMARK_NAME_BY_INDEX,
    PixelLandmark,
    BodyRegionStatus,
    aggregate_region_summaries,
    calculate_bmi,
    derive_body_metrics,
    summarize_analyzed_regions,
)


def _landmark(index: int, x_px: float, y_px: float, visibility: float = 0.99) -> PixelLandmark:
    return PixelLandmark(
        index=index,
        name=LANDMARK_NAME_BY_INDEX[index],
        x_px=x_px,
        y_px=y_px,
        visibility=visibility,
        presence=0.99,
    )


def test_calculate_bmi_returns_expected_value() -> None:
    assert calculate_bmi(height_cm=180.0, weight_kg=81.0) == 25.0


def test_calculate_bmi_returns_none_when_inputs_missing() -> None:
    assert calculate_bmi(height_cm=None, weight_kg=81.0) is None
    assert calculate_bmi(height_cm=180.0, weight_kg=None) is None


def test_derive_body_metrics_returns_expected_px_estimates() -> None:
    landmarks = [
        _landmark(11, 100.0, 120.0),
        _landmark(12, 220.0, 120.0),
        _landmark(23, 110.0, 280.0),
        _landmark(24, 210.0, 280.0),
        _landmark(25, 120.0, 420.0),
        _landmark(26, 200.0, 420.0),
        _landmark(27, 125.0, 560.0),
        _landmark(28, 195.0, 560.0),
    ]

    metrics = derive_body_metrics(landmarks)

    assert metrics.estimated_shoulder_width_px == 120.0
    assert metrics.estimated_hip_width_px == 100.0
    assert metrics.estimated_waist_width_px == 109.0
    assert metrics.estimated_waist_to_hip_ratio == 1.09
    assert metrics.posture_summary == "upright / balanced"
    assert metrics.lower_body_visibility_score == 0.99
    assert metrics.warnings == []


def test_derive_body_metrics_returns_nulls_with_low_visibility() -> None:
    landmarks = [
        _landmark(11, 100.0, 120.0, visibility=0.2),
        _landmark(12, 220.0, 120.0, visibility=0.2),
        _landmark(23, 110.0, 280.0, visibility=0.2),
        _landmark(24, 210.0, 280.0, visibility=0.2),
    ]

    metrics = derive_body_metrics(landmarks)

    assert metrics.estimated_shoulder_width_px is None
    assert metrics.estimated_hip_width_px is None
    assert metrics.estimated_waist_width_px is None
    assert metrics.estimated_waist_to_hip_ratio is None
    assert metrics.posture_summary is None
    assert len(metrics.warnings) >= 3


def test_summarize_analyzed_regions_marks_visible_and_cropped_areas() -> None:
    landmarks = [
        _landmark(0, 160.0, 40.0),
        _landmark(7, 120.0, 55.0),
        _landmark(8, 200.0, 55.0),
        _landmark(11, 100.0, 120.0),
        _landmark(12, 220.0, 120.0),
        _landmark(13, 85.0, 180.0, visibility=0.82),
        _landmark(14, 235.0, 180.0),
        _landmark(15, 65.0, 250.0, visibility=0.71),
        _landmark(16, 255.0, 250.0, visibility=0.73),
        _landmark(23, 110.0, 280.0),
        _landmark(24, 210.0, 280.0),
        _landmark(25, 120.0, 585.0, visibility=0.92),
        _landmark(26, 200.0, 586.0, visibility=0.9),
        _landmark(27, 125.0, 596.0, visibility=0.89),
        _landmark(28, 195.0, 598.0, visibility=0.91),
    ]

    regions = {
        region.key: region
        for region in summarize_analyzed_regions(landmarks, image_width=320, image_height=600)
    }

    assert regions["head"].visible is True
    assert regions["left_upper_arm"].visible is True
    assert regions["left_forearm"].visible is True
    assert regions["left_forearm"].confidence > 0.75
    assert regions["left_forearm"].taken_into_account is True
    assert regions["left_lower_leg"].visible is False
    assert regions["right_lower_leg"].visible is False
    assert regions["left_lower_leg"].confidence < 0.3
    assert regions["left_lower_leg"].taken_into_account is False


def test_summarize_analyzed_regions_rejects_thighs_when_knees_are_weak() -> None:
    landmarks = [
        _landmark(11, 807.57, 920.01),
        _landmark(12, 295.45, 924.36),
        _landmark(23, 683.38, 1643.11),
        _landmark(24, 367.32, 1619.47),
        _landmark(25, 714.07, 2152.3, visibility=0.3947),
        _landmark(26, 360.38, 2113.73, visibility=0.4064),
        _landmark(27, 646.6, 2424.0, visibility=0.0423),
        _landmark(28, 341.22, 2424.0, visibility=0.0462),
    ]

    regions = {
        region.key: region
        for region in summarize_analyzed_regions(landmarks, image_width=1080, image_height=2424)
    }

    assert regions["left_thigh"].visible is False
    assert regions["right_thigh"].visible is False
    assert regions["left_thigh"].confidence < 0.5
    assert regions["right_thigh"].confidence < 0.5


def test_summarize_analyzed_regions_rejects_thighs_without_downstream_leg_support() -> None:
    landmarks = [
        _landmark(11, 807.57, 920.01),
        _landmark(12, 295.45, 924.36),
        _landmark(23, 683.38, 1643.11),
        _landmark(24, 367.32, 1619.47),
        _landmark(25, 714.07, 2152.3, visibility=0.93),
        _landmark(26, 360.38, 2113.73, visibility=0.91),
        PixelLandmark(27, LANDMARK_NAME_BY_INDEX[27], 646.6, 2424.0, visibility=0.89, presence=0.08),
        PixelLandmark(28, LANDMARK_NAME_BY_INDEX[28], 341.22, 2424.0, visibility=0.87, presence=0.07),
        PixelLandmark(29, LANDMARK_NAME_BY_INDEX[29], 634.74, 2424.0, visibility=0.85, presence=0.05),
        PixelLandmark(30, LANDMARK_NAME_BY_INDEX[30], 335.44, 2424.0, visibility=0.84, presence=0.04),
        PixelLandmark(31, LANDMARK_NAME_BY_INDEX[31], 664.19, 2424.0, visibility=0.82, presence=0.03),
        PixelLandmark(32, LANDMARK_NAME_BY_INDEX[32], 369.88, 2424.0, visibility=0.81, presence=0.02),
    ]

    regions = {
        region.key: region
        for region in summarize_analyzed_regions(landmarks, image_width=1080, image_height=2424)
    }

    assert regions["left_thigh"].visible is False
    assert regions["right_thigh"].visible is False
    assert regions["left_lower_leg"].visible is False
    assert regions["right_lower_leg"].visible is False


def test_aggregate_region_summaries_returns_frontend_friendly_top_level_view() -> None:
    aggregated = {
        region.key: region
        for region in aggregate_region_summaries(
            [
                [
                    BodyRegionStatus(
                        key="head",
                        label="Tete",
                        visible=True,
                        confidence=0.91,
                        taken_into_account=True,
                    ),
                    BodyRegionStatus(
                        key="left_forearm",
                        label="Avant-bras gauche",
                        visible=True,
                        confidence=0.54,
                        taken_into_account=False,
                    ),
                ],
                [
                    BodyRegionStatus(
                        key="head",
                        label="Tete",
                        visible=True,
                        confidence=0.88,
                        taken_into_account=True,
                    ),
                    BodyRegionStatus(
                        key="left_forearm",
                        label="Avant-bras gauche",
                        visible=True,
                        confidence=0.58,
                        taken_into_account=True,
                    ),
                ],
            ]
        )
    }

    assert aggregated["head"].confidence == 0.91
    assert aggregated["head"].taken_into_account is True
    assert aggregated["left_forearm"].confidence == 0.58
    assert aggregated["left_forearm"].visible is True
    assert aggregated["left_forearm"].taken_into_account is True
