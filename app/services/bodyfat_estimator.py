"""Heuristic body fat estimator built on demographics and visual proxies."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.body_metrics import calculate_bmi
from app.services.feature_engineering import AggregatedVisualFeatures


SUPPORTED_SEX_VALUES = {
    "male": "male",
    "man": "male",
    "m": "male",
    "female": "female",
    "woman": "female",
    "f": "female",
}


@dataclass(frozen=True, slots=True)
class BodyFatEstimate:
    """Result of the top-level body fat inference."""

    bmi: float | None
    estimated_body_fat_percent: float | None
    estimated_fat_mass_kg: float | None
    estimated_lean_mass_kg: float | None
    confidence_score: float | None
    warnings: list[str]
    model_version: str


@dataclass(frozen=True, slots=True)
class SomatotypeEstimate:
    """Heuristic somatotype estimate derived from body scan proxies."""

    primary: str | None
    secondary: str | None
    confidence: float | None
    ectomorph_score: float
    mesomorph_score: float
    endomorph_score: float
    notes: str


def estimate_body_fat(
    *,
    age: int | None,
    sex: str | None,
    height_cm: float | None,
    weight_kg: float | None,
    visual_features: AggregatedVisualFeatures,
    model_version: str,
) -> BodyFatEstimate:
    """Estimate body fat from demographics, BMI, and image-derived proxies."""

    warnings = list(visual_features.warnings)
    warnings.append(
        "Estimated body fat is derived from demographics and image-based proxies, not direct measurement."
    )

    bmi = calculate_bmi(height_cm=height_cm, weight_kg=weight_kg)
    canonical_sex = normalize_sex(sex)
    if canonical_sex is None:
        warnings.append("Sex must be provided as male or female for body fat estimation.")
    if age is None:
        warnings.append("Age is required for body fat estimation.")
    if bmi is None:
        warnings.append("Height and weight are required to compute BMI and body fat estimation.")
    if visual_features.usable_image_count == 0:
        warnings.append("Reliable visual features are required for body fat estimation.")

    if canonical_sex is None or age is None or bmi is None or visual_features.usable_image_count == 0:
        return BodyFatEstimate(
            bmi=bmi,
            estimated_body_fat_percent=None,
            estimated_fat_mass_kg=None,
            estimated_lean_mass_kg=None,
            confidence_score=_round_or_none(_estimate_low_confidence(visual_features)),
            warnings=_deduplicate_strings(warnings),
            model_version=model_version,
        )

    confidence_score = _compute_confidence_score(
        canonical_sex=canonical_sex,
        age=age,
        bmi=bmi,
        visual_features=visual_features,
    )
    if not _has_sufficient_visual_support(visual_features):
        warnings.append("Visual support is too weak for a stable body fat estimate.")
        return BodyFatEstimate(
            bmi=bmi,
            estimated_body_fat_percent=None,
            estimated_fat_mass_kg=None,
            estimated_lean_mass_kg=None,
            confidence_score=confidence_score,
            warnings=_deduplicate_strings(warnings),
            model_version=model_version,
        )

    visual_estimate = _estimate_from_visual_features(visual_features, canonical_sex)
    bmi_prior = _estimate_from_bmi_prior(age=age, sex=canonical_sex, bmi=bmi)
    bmi_weight = _compute_bmi_weight(visual_features, canonical_sex)
    estimated_body_fat_percent = (visual_estimate * (1.0 - bmi_weight)) + (bmi_prior * bmi_weight)
    estimated_body_fat_percent += _visual_adjustment(visual_features, canonical_sex)
    estimated_body_fat_percent = round(_clamp(estimated_body_fat_percent, 3.0, 65.0), 2)

    if confidence_score is not None and confidence_score < 0.45:
        warnings.append("Input quality is weak, so the body fat estimate has low confidence.")

    fat_mass_kg = None
    lean_mass_kg = None
    if weight_kg is not None:
        fat_mass_kg = round(weight_kg * (estimated_body_fat_percent / 100.0), 2)
        lean_mass_kg = round(max(0.0, weight_kg - fat_mass_kg), 2)

    return BodyFatEstimate(
        bmi=bmi,
        estimated_body_fat_percent=estimated_body_fat_percent,
        estimated_fat_mass_kg=fat_mass_kg,
        estimated_lean_mass_kg=lean_mass_kg,
        confidence_score=confidence_score,
        warnings=_deduplicate_strings(warnings),
        model_version=model_version,
    )


def normalize_sex(value: str | None) -> str | None:
    """Normalize user-provided sex/gender values for the estimator."""

    if value is None:
        return None
    normalized = value.strip().lower()
    return SUPPORTED_SEX_VALUES.get(normalized)


def estimate_somatotype(
    *,
    sex: str | None,
    bmi: float | None,
    visual_features: AggregatedVisualFeatures,
    body_fat_percent: float | None,
) -> SomatotypeEstimate:
    """Estimate a rough somatotype profile from image-based proxies."""

    if visual_features.usable_image_count == 0 or visual_features.feature_richness_score < 0.3:
        return SomatotypeEstimate(
            primary=None,
            secondary=None,
            confidence=0.0,
            ectomorph_score=0.0,
            mesomorph_score=0.0,
            endomorph_score=0.0,
            notes=(
                "Les donnees visuelles disponibles ne sont pas encore assez fiables pour estimer un somatotype."
            ),
        )

    waist_to_bbox = visual_features.estimated_waist_to_bbox_height_ratio
    shoulder_to_hip = visual_features.estimated_shoulder_to_hip_ratio
    body_coverage = visual_features.body_coverage_score or 0.0
    canonical_sex = normalize_sex(sex)

    ectomorph = 0.0
    if bmi is not None:
        ectomorph += _inverse_normalized_value(bmi, 19.0, 27.0) or 0.0
    ectomorph += _inverse_normalized_value(body_fat_percent, 10.0, 28.0) or 0.0
    ectomorph += _inverse_normalized_value(waist_to_bbox, 0.16, 0.27) or 0.0
    ectomorph = ectomorph / 3.0

    mesomorph = 0.0
    mesomorph += _normalized_value(shoulder_to_hip, 0.95, 1.3) or 0.0
    mesomorph += _inverse_normalized_value(body_fat_percent, 12.0, 26.0) or 0.0
    mesomorph += _normalized_value(body_coverage, 0.45, 0.95) or 0.0
    mesomorph = mesomorph / 3.0

    endomorph = 0.0
    if bmi is not None:
        endomorph += _normalized_value(bmi, 21.0, 31.0) or 0.0
    endomorph += _normalized_value(body_fat_percent, 12.0, 34.0) or 0.0
    endomorph += _normalized_value(waist_to_bbox, 0.17, 0.30) or 0.0
    endomorph = endomorph / 3.0

    if canonical_sex == "male" and shoulder_to_hip is not None and shoulder_to_hip >= 1.18:
        mesomorph = min(1.0, mesomorph + 0.06)
    if canonical_sex == "female" and body_fat_percent is not None and body_fat_percent <= 23.0:
        ectomorph = min(1.0, ectomorph + 0.04)

    rounded_scores = {
        "ectomorph": round(_clamp(ectomorph, 0.0, 1.0), 3),
        "mesomorph": round(_clamp(mesomorph, 0.0, 1.0), 3),
        "endomorph": round(_clamp(endomorph, 0.0, 1.0), 3),
    }
    ordered = sorted(rounded_scores.items(), key=lambda item: item[1], reverse=True)
    confidence = round(_clamp((ordered[0][1] - ordered[1][1]) + (body_coverage * 0.35), 0.0, 1.0), 3)

    return SomatotypeEstimate(
        primary=ordered[0][0],
        secondary=ordered[1][0],
        confidence=confidence,
        ectomorph_score=rounded_scores["ectomorph"],
        mesomorph_score=rounded_scores["mesomorph"],
        endomorph_score=rounded_scores["endomorph"],
        notes=(
            "Estime a partir des proportions visibles du corps et d'heuristiques liees au body fat uniquement. "
            "Il s'agit d'une etiquette approximative et non medicale."
        ),
    )


def _visual_adjustment(visual_features: AggregatedVisualFeatures, canonical_sex: str) -> float:
    central_fat = visual_features.central_fat_score
    definition = visual_features.definition_score
    frame = visual_features.frame_score

    adjustment = 0.0
    if central_fat is not None:
        adjustment += (central_fat - 0.5) * 2.8
    if definition is not None:
        adjustment -= (definition - 0.5) * 2.2
    if frame is not None:
        adjustment -= max(0.0, frame - 0.62) * (0.9 if canonical_sex == "male" else 0.45)
    return _clamp(adjustment, -2.5, 2.5)


def _estimate_from_visual_features(
    visual_features: AggregatedVisualFeatures,
    canonical_sex: str,
) -> float:
    if canonical_sex == "male":
        score = _weighted_component_score(
            (
                _normalized_value(visual_features.estimated_waist_to_hip_ratio, 0.78, 1.02),
                0.45,
            ),
            (
                _normalized_value(visual_features.estimated_waist_to_bbox_height_ratio, 0.14, 0.28),
                0.35,
            ),
            (
                _inverse_normalized_value(visual_features.estimated_shoulder_to_hip_ratio, 0.95, 1.35),
                0.20,
            ),
        )
        estimate = 7.0 + (score * 20.0)
        if _looks_very_lean(visual_features, canonical_sex):
            estimate -= 1.25
        if (visual_features.definition_score or 0.0) >= 0.78:
            estimate -= 0.6
        return estimate

    score = _weighted_component_score(
        (
            _normalized_value(visual_features.estimated_waist_to_hip_ratio, 0.68, 0.92),
            0.45,
        ),
        (
            _normalized_value(visual_features.estimated_waist_to_bbox_height_ratio, 0.16, 0.31),
            0.35,
        ),
        (
            _inverse_normalized_value(visual_features.estimated_shoulder_to_hip_ratio, 0.88, 1.12),
            0.20,
        ),
    )
    estimate = 15.0 + (score * 21.0)
    if _looks_very_lean(visual_features, canonical_sex):
        estimate -= 1.0
    if (visual_features.definition_score or 0.0) >= 0.78:
        estimate -= 0.4
    return estimate


def _estimate_from_bmi_prior(*, age: int, sex: str, bmi: float) -> float:
    if sex == "male":
        return (1.02 * bmi) + (0.10 * age) - 14.8
    return (1.04 * bmi) + (0.09 * age) - 6.5


def _compute_bmi_weight(visual_features: AggregatedVisualFeatures, canonical_sex: str) -> float:
    weight = 0.22
    if visual_features.usable_image_count >= 2:
        weight -= 0.05
    if (visual_features.feature_consistency_score or 0.0) >= 0.75:
        weight -= 0.04
    if _looks_very_lean(visual_features, canonical_sex):
        weight -= 0.05
    if (visual_features.definition_score or 0.0) >= 0.72:
        weight -= 0.03
    return _clamp(weight, 0.08, 0.24)


def _has_sufficient_visual_support(visual_features: AggregatedVisualFeatures) -> bool:
    if visual_features.usable_image_count == 0:
        return False
    if visual_features.feature_richness_score < 0.4:
        return False
    if (visual_features.aggregate_quality_score or 0.0) < 0.45:
        return False
    return True


def _looks_very_lean(visual_features: AggregatedVisualFeatures, canonical_sex: str) -> bool:
    waist_to_hip = visual_features.estimated_waist_to_hip_ratio
    waist_to_bbox = visual_features.estimated_waist_to_bbox_height_ratio
    shoulder_to_hip = visual_features.estimated_shoulder_to_hip_ratio

    if canonical_sex == "male":
        return (
            waist_to_hip is not None and waist_to_hip <= 0.86
            and waist_to_bbox is not None and waist_to_bbox <= 0.185
            and shoulder_to_hip is not None and shoulder_to_hip >= 1.18
        )
    return (
        waist_to_hip is not None and waist_to_hip <= 0.74
        and waist_to_bbox is not None and waist_to_bbox <= 0.205
        and shoulder_to_hip is not None and shoulder_to_hip >= 0.98
    )


def _compute_confidence_score(
    *,
    canonical_sex: str,
    age: int,
    bmi: float,
    visual_features: AggregatedVisualFeatures,
) -> float | None:
    del canonical_sex, age, bmi

    if visual_features.reliability_score is not None:
        return round(_clamp(visual_features.reliability_score, 0.0, 1.0), 3)

    feature_richness = visual_features.feature_richness_score
    image_support = min(1.0, visual_features.usable_image_count / 3.0)
    aggregate_quality = visual_features.aggregate_quality_score or 0.0
    consistency = visual_features.feature_consistency_score or 0.0
    body_coverage = visual_features.body_coverage_score or 0.0
    confidence = (
        (aggregate_quality * 0.25)
        + (feature_richness * 0.20)
        + (image_support * 0.15)
        + (consistency * 0.15)
        + (body_coverage * 0.25)
    )
    return round(_clamp(confidence, 0.0, 1.0), 3)


def _estimate_low_confidence(visual_features: AggregatedVisualFeatures) -> float | None:
    if visual_features.total_image_count == 0:
        return 0.0
    return min(0.35, ((visual_features.aggregate_quality_score or 0.0) * 0.35) + 0.05)


def _weighted_component_score(*pairs: tuple[float | None, float]) -> float:
    usable_pairs = [(value, weight) for value, weight in pairs if value is not None]
    if not usable_pairs:
        return 0.5
    weighted_sum = sum(value * weight for value, weight in usable_pairs)
    total_weight = sum(weight for _, weight in usable_pairs)
    return weighted_sum / total_weight


def _normalized_value(value: float | None, low: float, high: float) -> float | None:
    if value is None:
        return None
    return _clamp((value - low) / (high - low), 0.0, 1.0)


def _inverse_normalized_value(value: float | None, low: float, high: float) -> float | None:
    normalized = _normalized_value(value, low, high)
    if normalized is None:
        return None
    return 1.0 - normalized


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def _deduplicate_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
