"""Pure functions for body-derived metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

FRONTEND_REGION_CONFIDENCE_THRESHOLD = 0.55

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

LANDMARK_NAME_BY_INDEX: dict[int, str] = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


@dataclass(frozen=True, slots=True)
class PixelLandmark:
    """Pose landmark in pixel coordinates."""

    index: int
    name: str
    x_px: float
    y_px: float
    visibility: float | None = None
    presence: float | None = None


@dataclass(frozen=True, slots=True)
class BodyRegionStatus:
    """Visibility summary for a coarse body region."""

    key: str
    label: str
    visible: bool
    confidence: float
    taken_into_account: bool


@dataclass(frozen=True, slots=True)
class DerivedBodyMetrics:
    """Estimated body metrics derived from pose landmarks."""

    estimated_shoulder_width_px: float | None
    estimated_hip_width_px: float | None
    estimated_waist_width_px: float | None
    estimated_waist_to_hip_ratio: float | None
    posture_summary: str | None
    lower_body_visibility_score: float | None
    warnings: list[str]


def calculate_bmi(height_cm: float | None, weight_kg: float | None) -> float | None:
    """Compute BMI when both height and weight are available and valid."""

    if height_cm is None or weight_kg is None:
        return None
    if height_cm <= 0 or weight_kg <= 0:
        return None
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m * height_m)
    return round(bmi, 2)


def derive_body_metrics(
    landmarks: list[PixelLandmark],
    visibility_threshold: float = 0.5,
) -> DerivedBodyMetrics:
    """Compute derived image-based metrics from visible landmarks."""

    warnings: list[str] = []
    landmark_map = {landmark.index: landmark for landmark in landmarks}

    shoulder_width = _distance_if_visible(
        landmark_map, LEFT_SHOULDER, RIGHT_SHOULDER, visibility_threshold
    )
    if shoulder_width is None:
        warnings.append("Shoulder width estimate is unreliable because shoulder landmarks are weak or missing.")

    hip_width = _distance_if_visible(landmark_map, LEFT_HIP, RIGHT_HIP, visibility_threshold)
    if hip_width is None:
        warnings.append("Hip width estimate is unreliable because hip landmarks are weak or missing.")

    waist_width = _estimate_waist_width(landmark_map, visibility_threshold)
    if waist_width is None:
        warnings.append("Waist width estimate is unavailable because torso landmarks are insufficient.")

    waist_to_hip_ratio = None
    if waist_width is not None and hip_width is not None and hip_width > 0:
        waist_to_hip_ratio = round(waist_width / hip_width, 3)
    elif waist_width is not None:
        warnings.append("Waist-to-hip ratio could not be computed because hip width is unavailable.")

    posture_summary = _summarize_posture(landmark_map, visibility_threshold)
    if posture_summary is None:
        warnings.append("Posture summary is unavailable because the torso landmarks are insufficient.")

    lower_body_visibility_score = _estimate_lower_body_visibility(landmark_map)
    if lower_body_visibility_score is None or lower_body_visibility_score < 0.45:
        warnings.append("Lower body is partially visible or cropped; confidence should be reduced.")

    return DerivedBodyMetrics(
        estimated_shoulder_width_px=_round_or_none(shoulder_width),
        estimated_hip_width_px=_round_or_none(hip_width),
        estimated_waist_width_px=_round_or_none(waist_width),
        estimated_waist_to_hip_ratio=waist_to_hip_ratio,
        posture_summary=posture_summary,
        lower_body_visibility_score=_round_or_none(lower_body_visibility_score),
        warnings=warnings,
    )


def summarize_analyzed_regions(
    landmarks: list[PixelLandmark],
    visibility_threshold: float = 0.5,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[BodyRegionStatus]:
    """Build a coarse checklist of visible body regions from pose landmarks."""

    landmark_map = {landmark.index: landmark for landmark in landmarks}
    torso_height = _estimate_torso_height(landmark_map)
    region_specs = [
        ("head", "Tete", [0, 7, 8], "average", False, []),
        ("left_upper_arm", "Biceps / triceps gauche", [11, 13], "arm_segment", False, []),
        ("right_upper_arm", "Biceps / triceps droit", [12, 14], "arm_segment", False, []),
        ("left_forearm", "Avant-bras gauche", [13, 15], "arm_segment", False, []),
        ("right_forearm", "Avant-bras droit", [14, 16], "arm_segment", False, []),
        ("torso", "Torse", [11, 12, 23, 24], "average", False, []),
        ("waist", "Taille", [11, 12, 23, 24], "average", False, []),
        ("hips", "Hanches", [23, 24], "average", False, []),
        (
            "left_thigh",
            "Cuisse gauche",
            [23, 25],
            "thigh_segment",
            True,
            [LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX],
        ),
        (
            "right_thigh",
            "Cuisse droite",
            [24, 26],
            "thigh_segment",
            True,
            [RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX],
        ),
        (
            "left_lower_leg",
            "Mollet / jambe gauche",
            [25, 27],
            "lower_leg_segment",
            True,
            [LEFT_HEEL, LEFT_FOOT_INDEX],
        ),
        (
            "right_lower_leg",
            "Mollet / jambe droite",
            [26, 28],
            "lower_leg_segment",
            True,
            [RIGHT_HEEL, RIGHT_FOOT_INDEX],
        ),
    ]
    return [
        _build_region_status(
            key=key,
            label=label,
            indices=indices,
            score_mode=score_mode,
            edge_sensitive=edge_sensitive,
            support_indices=support_indices,
            landmark_map=landmark_map,
            visibility_threshold=visibility_threshold,
            image_width=image_width,
            image_height=image_height,
            torso_height=torso_height,
        )
        for key, label, indices, score_mode, edge_sensitive, support_indices in region_specs
    ]


def aggregate_region_summaries(
    region_sets: list[list[BodyRegionStatus]],
    frontend_threshold: float = FRONTEND_REGION_CONFIDENCE_THRESHOLD,
) -> list[BodyRegionStatus]:
    """Aggregate region visibility across images for a frontend-friendly summary."""

    region_order: list[str] = []
    aggregates: dict[str, dict[str, str | bool | float]] = {}

    for region_set in region_sets:
        for region in region_set:
            if region.key not in aggregates:
                region_order.append(region.key)
                aggregates[region.key] = {
                    "label": region.label,
                    "visible": region.visible,
                    "confidence": region.confidence,
                }
                continue

            aggregate = aggregates[region.key]
            aggregate["visible"] = bool(aggregate["visible"]) or region.visible
            aggregate["confidence"] = max(float(aggregate["confidence"]), region.confidence)

    return [
        BodyRegionStatus(
            key=key,
            label=str(aggregates[key]["label"]),
            visible=bool(aggregates[key]["visible"]),
            confidence=round(float(aggregates[key]["confidence"]), 3),
            taken_into_account=float(aggregates[key]["confidence"]) >= frontend_threshold,
        )
        for key in region_order
    ]


def _estimate_waist_width(
    landmark_map: dict[int, PixelLandmark],
    visibility_threshold: float,
) -> float | None:
    left_shoulder = landmark_map.get(LEFT_SHOULDER)
    right_shoulder = landmark_map.get(RIGHT_SHOULDER)
    left_hip = landmark_map.get(LEFT_HIP)
    right_hip = landmark_map.get(RIGHT_HIP)

    required = [left_shoulder, right_shoulder, left_hip, right_hip]
    if any(landmark is None for landmark in required):
        return None
    if any(not _is_reliable(landmark, visibility_threshold) for landmark in required):
        return None

    left_proxy = _interpolate(left_shoulder, left_hip, 0.55)
    right_proxy = _interpolate(right_shoulder, right_hip, 0.55)
    return _distance(left_proxy, right_proxy)


def _summarize_posture(
    landmark_map: dict[int, PixelLandmark],
    visibility_threshold: float,
) -> str | None:
    left_shoulder = landmark_map.get(LEFT_SHOULDER)
    right_shoulder = landmark_map.get(RIGHT_SHOULDER)
    left_hip = landmark_map.get(LEFT_HIP)
    right_hip = landmark_map.get(RIGHT_HIP)

    required = [left_shoulder, right_shoulder, left_hip, right_hip]
    if any(landmark is None for landmark in required):
        return None
    if any(not _is_reliable(landmark, visibility_threshold) for landmark in required):
        return None

    shoulder_width = _distance(left_shoulder, right_shoulder)
    hip_width = _distance(left_hip, right_hip)
    if shoulder_width == 0 or hip_width == 0:
        return None

    shoulder_tilt = abs(left_shoulder.y_px - right_shoulder.y_px) / shoulder_width
    hip_tilt = abs(left_hip.y_px - right_hip.y_px) / hip_width
    shoulder_mid_x = (left_shoulder.x_px + right_shoulder.x_px) / 2.0
    hip_mid_x = (left_hip.x_px + right_hip.x_px) / 2.0
    lateral_offset = abs(shoulder_mid_x - hip_mid_x) / hip_width

    if shoulder_tilt < 0.08 and hip_tilt < 0.08 and lateral_offset < 0.12:
        return "upright / balanced"
    if lateral_offset >= 0.18:
        return "possible lateral lean"
    if shoulder_tilt >= 0.12 or hip_tilt >= 0.12:
        return "possible tilt / asymmetry"
    return "slight asymmetry"


def _distance_if_visible(
    landmark_map: dict[int, PixelLandmark],
    first_index: int,
    second_index: int,
    visibility_threshold: float,
) -> float | None:
    first = landmark_map.get(first_index)
    second = landmark_map.get(second_index)
    if first is None or second is None:
        return None
    if not _is_reliable(first, visibility_threshold) or not _is_reliable(second, visibility_threshold):
        return None
    return _distance(first, second)


def _estimate_lower_body_visibility(landmark_map: dict[int, PixelLandmark]) -> float | None:
    lower_body_indices = [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    visible_scores = [
        _landmark_signal(landmark_map[index])
        for index in lower_body_indices
        if index in landmark_map
    ]
    if not visible_scores:
        return None
    return sum(visible_scores) / len(visible_scores)


def _build_region_status(
    *,
    key: str,
    label: str,
    indices: list[int],
    score_mode: str,
    edge_sensitive: bool,
    support_indices: list[int],
    landmark_map: dict[int, PixelLandmark],
    visibility_threshold: float,
    image_width: int | None,
    image_height: int | None,
    torso_height: float | None,
) -> BodyRegionStatus:
    region_landmarks = [landmark_map[index] for index in indices if index in landmark_map]
    if not region_landmarks:
        return BodyRegionStatus(
            key=key,
            label=label,
            visible=False,
            confidence=0.0,
            taken_into_account=False,
        )

    visible_scores = [_landmark_signal(landmark) for landmark in region_landmarks]
    confidence = _score_region_confidence(
        region_landmarks=region_landmarks,
        visible_scores=visible_scores,
        score_mode=score_mode,
        torso_height=torso_height,
        visibility_threshold=visibility_threshold,
        support_landmarks=[landmark_map[index] for index in support_indices if index in landmark_map],
    )
    if edge_sensitive:
        confidence *= _compute_edge_penalty(
            region_landmarks=region_landmarks,
            image_width=image_width,
            image_height=image_height,
        )
    confidence = round(min(1.0, max(0.0, confidence)), 3)
    return BodyRegionStatus(
        key=key,
        label=label,
        visible=confidence >= visibility_threshold,
        confidence=confidence,
        taken_into_account=confidence >= FRONTEND_REGION_CONFIDENCE_THRESHOLD,
    )


def _score_region_confidence(
    *,
    region_landmarks: list[PixelLandmark],
    visible_scores: list[float],
    score_mode: str,
    torso_height: float | None,
    visibility_threshold: float,
    support_landmarks: list[PixelLandmark],
) -> float:
    average_score = sum(visible_scores) / len(visible_scores)
    if score_mode == "arm_segment":
        max_score = max(visible_scores)
        geometric_score = _segment_geometry_score(
            first=region_landmarks[0],
            second=region_landmarks[-1],
            torso_height=torso_height,
            min_vertical_ratio=0.08,
            max_vertical_ratio=1.05,
        )
        return min(1.0, ((average_score * 0.35) + (max_score * 0.45) + (geometric_score * 0.20)))
    if score_mode == "thigh_segment":
        segment_score = _segment_geometry_score(
            first=region_landmarks[0],
            second=region_landmarks[-1],
            torso_height=torso_height,
            min_vertical_ratio=0.55,
            max_vertical_ratio=1.8,
        )
        distal_gate = _distal_visibility_gate(region_landmarks[-1], visibility_threshold)
        support_gate = _downstream_support_gate(support_landmarks)
        return _distal_emphasis_score(visible_scores) * segment_score * distal_gate * support_gate
    if score_mode == "lower_leg_segment":
        segment_score = _segment_geometry_score(
            first=region_landmarks[0],
            second=region_landmarks[-1],
            torso_height=torso_height,
            min_vertical_ratio=0.45,
            max_vertical_ratio=1.7,
        )
        distal_gate = _distal_visibility_gate(region_landmarks[-1], visibility_threshold)
        support_gate = _downstream_support_gate(support_landmarks)
        return _distal_emphasis_score(visible_scores) * segment_score * distal_gate * support_gate
    return average_score


def _segment_geometry_score(
    *,
    first: PixelLandmark,
    second: PixelLandmark,
    torso_height: float | None,
    min_vertical_ratio: float,
    max_vertical_ratio: float,
) -> float:
    if torso_height is None or torso_height <= 0:
        return 0.4

    vertical_delta = second.y_px - first.y_px
    if vertical_delta <= 0:
        return 0.0

    vertical_ratio = vertical_delta / torso_height
    if vertical_ratio < min_vertical_ratio:
        return max(0.0, vertical_ratio / min_vertical_ratio)
    if vertical_ratio > max_vertical_ratio:
        overflow = vertical_ratio - max_vertical_ratio
        return max(0.0, 1.0 - (overflow / max_vertical_ratio))
    return 1.0


def _distal_emphasis_score(visible_scores: list[float]) -> float:
    if not visible_scores:
        return 0.0
    if len(visible_scores) == 1:
        return visible_scores[0]
    proximal_score = visible_scores[0]
    distal_score = visible_scores[-1]
    return (proximal_score * 0.15) + (distal_score * 0.85)


def _distal_visibility_gate(landmark: PixelLandmark, visibility_threshold: float) -> float:
    signal = _landmark_signal(landmark)
    if visibility_threshold <= 0:
        return 1.0
    return min(1.0, signal / visibility_threshold)


def _downstream_support_gate(support_landmarks: list[PixelLandmark]) -> float:
    if not support_landmarks:
        return 0.2
    support_signal = max(_landmark_signal(landmark) for landmark in support_landmarks)
    if support_signal >= 0.5:
        return 1.0
    if support_signal >= 0.35:
        return 0.6
    if support_signal >= 0.2:
        return 0.3
    return 0.12


def _compute_edge_penalty(
    *,
    region_landmarks: list[PixelLandmark],
    image_width: int | None,
    image_height: int | None,
) -> float:
    if image_width is None or image_height is None:
        return 1.0

    horizontal_margin = image_width * 0.03
    vertical_margin = image_height * 0.04
    min_penalty = 1.0

    for landmark in region_landmarks:
        penalties = []
        if landmark.y_px >= image_height - vertical_margin:
            penalties.append(0.18)
        elif landmark.y_px >= image_height - (vertical_margin * 2.0):
            penalties.append(0.45)
        if landmark.x_px <= horizontal_margin or landmark.x_px >= image_width - horizontal_margin:
            penalties.append(0.6)
        if penalties:
            min_penalty = min(min_penalty, min(penalties))

    return min_penalty


def _estimate_torso_height(landmark_map: dict[int, PixelLandmark]) -> float | None:
    left_shoulder = landmark_map.get(LEFT_SHOULDER)
    right_shoulder = landmark_map.get(RIGHT_SHOULDER)
    left_hip = landmark_map.get(LEFT_HIP)
    right_hip = landmark_map.get(RIGHT_HIP)
    required = [left_shoulder, right_shoulder, left_hip, right_hip]
    if any(landmark is None for landmark in required):
        return None

    shoulder_mid_y = (left_shoulder.y_px + right_shoulder.y_px) / 2.0
    hip_mid_y = (left_hip.y_px + right_hip.y_px) / 2.0
    torso_height = hip_mid_y - shoulder_mid_y
    if torso_height <= 0:
        return None
    return torso_height


def _interpolate(first: PixelLandmark, second: PixelLandmark, ratio: float) -> PixelLandmark:
    return PixelLandmark(
        index=-1,
        name="proxy",
        x_px=first.x_px + (second.x_px - first.x_px) * ratio,
        y_px=first.y_px + (second.y_px - first.y_px) * ratio,
        visibility=min(_visibility_or_zero(first), _visibility_or_zero(second)),
        presence=min(_presence_or_zero(first), _presence_or_zero(second)),
    )


def _distance(first: PixelLandmark, second: PixelLandmark) -> float:
    return sqrt((first.x_px - second.x_px) ** 2 + (first.y_px - second.y_px) ** 2)


def _is_reliable(landmark: PixelLandmark, visibility_threshold: float) -> bool:
    return _visibility_or_zero(landmark) >= visibility_threshold


def _visibility_or_zero(landmark: PixelLandmark) -> float:
    return float(landmark.visibility or 0.0)


def _presence_or_zero(landmark: PixelLandmark) -> float:
    return float(landmark.presence or 0.0)


def _landmark_signal(landmark: PixelLandmark) -> float:
    return min(_visibility_or_zero(landmark), _presence_or_zero(landmark))


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 2)
