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


@dataclass(frozen=True, slots=True)
class AnalysisNote:
    """Single frontend-oriented note with title and message."""

    title: str
    message: str


@dataclass(frozen=True, slots=True)
class AnalysisNotes:
    """Three frontend-oriented notes for the product mockup."""

    attention: AnalysisNote
    parfait: AnalysisNote
    progression: AnalysisNote


@dataclass(frozen=True, slots=True)
class AnalysisBlocks:
    """Expanded premium cards for a richer frontend analysis experience."""

    overview: AnalysisNote
    truth: AnalysisNote
    strength: AnalysisNote
    limitation: AnalysisNote
    next_focus: AnalysisNote
    scan_quality: AnalysisNote


@dataclass(frozen=True, slots=True)
class ScanProfile:
    """Structured premium profile derived from the scan."""

    reliability_level: str
    confidence_label: str
    confidence_message: str
    definition_level: str
    fat_distribution: str
    frame_assessment: str
    view_coverage: str
    pose_quality: str
    scan_readiness: str
    best_next_focus: str
    dominant_strength: str
    dominant_limitation: str
    summary: str


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
    sex: str | None = None,
    visual_features: AggregatedVisualFeatures | None = None,
) -> str:
    """Generate a short, readable summary of what looked usable in the analysis."""

    if not has_successful_detection:
        return "Aucun corps n'a ete detecte de maniere suffisamment fiable. Reprends au moins une photo nette et bien cadree."

    strong_regions = [region.label for region in analyzed_regions_summary if region.taken_into_account]
    weak_regions = [region.label for region in analyzed_regions_summary if not region.taken_into_account]

    reliability = visual_features.reliability_score if visual_features is not None else None
    effective_confidence = confidence_score if confidence_score is not None else reliability

    if effective_confidence is not None and effective_confidence >= 0.82:
        opening = "La base d'analyse est vraiment solide."
    elif effective_confidence is not None and effective_confidence >= 0.6:
        opening = "L'analyse est exploitable, avec encore quelques zones a confirmer."
    else:
        opening = "L'analyse reste utile, mais la confiance globale est encore moyenne."

    if overall_quality_score is not None and overall_quality_score >= 0.75:
        quality_note = "La qualite des images est globalement bonne."
    elif overall_quality_score is not None and overall_quality_score >= 0.55:
        quality_note = "La qualite des images est correcte, sans etre optimale."
    else:
        quality_note = "La qualite des images limite encore l'analyse."

    estimate_note = _build_body_status_note(
        sex=sex,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
    )

    region_notes: list[str] = []
    if strong_regions:
        region_notes.append(f"Zones les plus fiables : {_join_labels(strong_regions[:4])}.")
    if weak_regions:
        region_notes.append(f"Zones encore faibles ou partielles : {_join_labels(weak_regions[:4])}.")

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
            "Les donnees actuelles ne sont pas encore assez solides pour proposer une projection de progression fiable. "
            "Utilise des photos full body plus propres et renseigne age, sex, height_cm et weight_kg."
        )

    if confidence_score is not None and confidence_score < 0.45:
        return (
            "La confiance reste trop faible pour proposer une projection de progression utile. "
            "Reprends des photos plus nettes et plus completes avant de te fier a une timeline."
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
        confidence_note = " La confiance reste moyenne, donc cette projection doit etre lue comme une tendance."

    return f"{composition_note} {trajectory_note}{confidence_note}".strip()


def build_analysis_notes(
    *,
    has_successful_detection: bool,
    overall_quality_score: float | None,
    confidence_score: float | None,
    estimated_body_fat_percent: float | None,
    estimated_lean_mass_kg: float | None,
    weight_kg: float | None,
    sex: str | None,
    analyzed_regions_summary: list[BodyRegionStatus],
    visual_features: AggregatedVisualFeatures,
    reference_date: date | None = None,
) -> AnalysisNotes:
    """Build the three product notes: attention, parfait, progression."""

    attention_message = _build_attention_note(
        has_successful_detection=has_successful_detection,
        overall_quality_score=overall_quality_score,
        confidence_score=confidence_score,
        analyzed_regions_summary=analyzed_regions_summary,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
        sex=sex,
    )
    parfait_message = _build_parfait_note(
        has_successful_detection=has_successful_detection,
        overall_quality_score=overall_quality_score,
        confidence_score=confidence_score,
        analyzed_regions_summary=analyzed_regions_summary,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
        sex=sex,
    )
    progression_message = _build_progression_note(
        sex=sex,
        confidence_score=confidence_score,
        estimated_body_fat_percent=estimated_body_fat_percent,
        estimated_lean_mass_kg=estimated_lean_mass_kg,
        weight_kg=weight_kg,
        visual_features=visual_features,
        analyzed_regions_summary=analyzed_regions_summary,
        reference_date=reference_date,
    )

    attention_title = _build_attention_title(
        has_successful_detection=has_successful_detection,
        overall_quality_score=overall_quality_score,
        confidence_score=confidence_score,
        analyzed_regions_summary=analyzed_regions_summary,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
        sex=sex,
    )
    parfait_title = _build_parfait_title(
        has_successful_detection=has_successful_detection,
        overall_quality_score=overall_quality_score,
        confidence_score=confidence_score,
        analyzed_regions_summary=analyzed_regions_summary,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
        sex=sex,
    )
    progression_title = _build_progression_title(
        sex=sex,
        confidence_score=confidence_score,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
        analyzed_regions_summary=analyzed_regions_summary,
    )

    return AnalysisNotes(
        attention=AnalysisNote(title=attention_title, message=attention_message),
        parfait=AnalysisNote(title=parfait_title, message=parfait_message),
        progression=AnalysisNote(title=progression_title, message=progression_message),
    )


def build_analysis_blocks(
    *,
    analysis_notes: AnalysisNotes,
    scan_profile: ScanProfile,
    confidence_score: float | None,
    overall_quality_score: float | None,
    estimated_body_fat_percent: float | None,
    sex: str | None,
) -> AnalysisBlocks:
    """Build a richer set of frontend-ready premium blocks."""

    return AnalysisBlocks(
        overview=AnalysisNote(
            title="Lecture globale",
            message=scan_profile.summary,
        ),
        truth=AnalysisNote(
            title=_build_truth_title(scan_profile=scan_profile),
            message=_build_truth_message(
                scan_profile=scan_profile,
                confidence_score=confidence_score,
                estimated_body_fat_percent=estimated_body_fat_percent,
                sex=sex,
            ),
        ),
        strength=AnalysisNote(
            title=_build_strength_title(scan_profile=scan_profile),
            message=_build_strength_message(scan_profile=scan_profile, analysis_notes=analysis_notes),
        ),
        limitation=AnalysisNote(
            title=_build_limitation_title(scan_profile=scan_profile),
            message=_build_limitation_message(scan_profile=scan_profile, analysis_notes=analysis_notes),
        ),
        next_focus=AnalysisNote(
            title=_build_next_focus_title(scan_profile=scan_profile),
            message=_build_next_focus_message(scan_profile=scan_profile, analysis_notes=analysis_notes),
        ),
        scan_quality=AnalysisNote(
            title="Niveau de fiabilite",
            message=_build_scan_quality_message(
                scan_profile=scan_profile,
                confidence_score=confidence_score,
                overall_quality_score=overall_quality_score,
            ),
        ),
    )


def build_scan_profile(
    *,
    has_successful_detection: bool,
    confidence_score: float | None,
    overall_quality_score: float | None,
    estimated_body_fat_percent: float | None,
    analyzed_regions_summary: list[BodyRegionStatus],
    visual_features: AggregatedVisualFeatures,
    sex: str | None,
) -> ScanProfile:
    """Build a premium, structured scan profile for the frontend."""

    weak_regions = [region.label for region in analyzed_regions_summary if not region.taken_into_account]
    effective_confidence = confidence_score if confidence_score is not None else visual_features.reliability_score

    reliability_level = _classify_reliability_level(
        has_successful_detection=has_successful_detection,
        effective_confidence=effective_confidence,
        overall_quality_score=overall_quality_score,
        weak_regions=weak_regions,
    )
    confidence_label = _classify_confidence_label(
        reliability_level=reliability_level,
        view_coverage=_classify_view_coverage(visual_features=visual_features),
        pose_quality=_classify_pose_quality(visual_features=visual_features),
    )
    definition_level = _classify_definition_level(
        sex=sex,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
    )
    fat_distribution = _classify_fat_distribution(
        sex=sex,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
    )
    frame_assessment = _classify_frame_assessment(
        sex=sex,
        visual_features=visual_features,
    )
    view_coverage = _classify_view_coverage(visual_features=visual_features)
    pose_quality = _classify_pose_quality(visual_features=visual_features)
    confidence_message = _build_confidence_message(
        reliability_level=reliability_level,
        view_coverage=view_coverage,
        pose_quality=pose_quality,
        weak_regions=weak_regions,
    )
    scan_readiness = _classify_scan_readiness(
        reliability_level=reliability_level,
        has_successful_detection=has_successful_detection,
        weak_regions=weak_regions,
    )
    best_next_focus = _classify_best_next_focus(
        reliability_level=reliability_level,
        definition_level=definition_level,
        fat_distribution=fat_distribution,
        weak_regions=weak_regions,
        view_coverage=view_coverage,
    )
    dominant_strength = _classify_dominant_strength(
        reliability_level=reliability_level,
        definition_level=definition_level,
        frame_assessment=frame_assessment,
        view_coverage=view_coverage,
        analyzed_regions_summary=analyzed_regions_summary,
    )
    dominant_limitation = _classify_dominant_limitation(
        reliability_level=reliability_level,
        fat_distribution=fat_distribution,
        definition_level=definition_level,
        weak_regions=weak_regions,
        view_coverage=view_coverage,
        pose_quality=pose_quality,
    )
    summary = _build_scan_profile_summary(
        reliability_level=reliability_level,
        definition_level=definition_level,
        fat_distribution=fat_distribution,
        frame_assessment=frame_assessment,
        view_coverage=view_coverage,
        pose_quality=pose_quality,
        best_next_focus=best_next_focus,
        dominant_strength=dominant_strength,
        dominant_limitation=dominant_limitation,
    )

    return ScanProfile(
        reliability_level=reliability_level,
        confidence_label=confidence_label,
        confidence_message=confidence_message,
        definition_level=definition_level,
        fat_distribution=fat_distribution,
        frame_assessment=frame_assessment,
        view_coverage=view_coverage,
        pose_quality=pose_quality,
        scan_readiness=scan_readiness,
        best_next_focus=best_next_focus,
        dominant_strength=dominant_strength,
        dominant_limitation=dominant_limitation,
        summary=summary,
    )


def _join_labels(labels: list[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} et {labels[1]}"
    return ", ".join(labels[:-1]) + f", et {labels[-1]}"


def _build_attention_note(
    *,
    has_successful_detection: bool,
    overall_quality_score: float | None,
    confidence_score: float | None,
    analyzed_regions_summary: list[BodyRegionStatus],
    estimated_body_fat_percent: float | None,
    visual_features: AggregatedVisualFeatures,
    sex: str | None,
) -> str:
    if not has_successful_detection:
        return "Attention: pour l'instant, aucune photo n'est assez propre pour dire quelque chose de fiable sur ton physique."

    weak_regions = [region.label for region in analyzed_regions_summary if not region.taken_into_account]
    if _is_upper_body_dominant_scan(analyzed_regions_summary=analyzed_regions_summary, visual_features=visual_features):
        return (
            "Attention: le haut du corps est bien lu, mais le bas du corps reste trop peu visible pour une analyse vraiment complete. "
            "On peut deja juger une partie du rendu, mais pas verrouiller tout le physique."
        )
    if confidence_score is not None and confidence_score < 0.45:
        return (
            "Attention: je ne peux pas etre strict sur le rendu du physique tant que la confiance reste aussi basse. "
            "Il faut d'abord reprendre des photos plus nettes."
        )
    if overall_quality_score is not None and overall_quality_score < 0.6:
        return (
            "Attention: la qualite photo bride encore l'analyse. Tant que le cadrage et la nettete ne montent pas, le diagnostic restera partiel. "
        )
    if (visual_features.view_diversity_score or 0.0) < 0.45 and visual_features.usable_image_count <= 1:
        return (
            "Attention: le scan repose encore sur un angle trop limite. Sans vraie vue complementaire, l'analyse reste moins dure qu'elle pourrait l'etre. "
        )
    if weak_regions:
        return (
            f"Attention: certaines zones restent trop fragiles pour une lecture vraiment stricte, surtout {_join_labels(weak_regions[:3])}."
        )

    if _has_abdominal_attention_signal(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent, visual_features=visual_features):
        if (sex or "").lower() == "male":
            return (
                "Attention: la verite qui ressort le plus clairement ici, c'est que l'abdomen reste encore la zone qui retient le plus de gras. "
                "Tant que cette zone ne descend pas, le rendu global restera moins sec."
            )
        if (sex or "").lower() == "female":
            return (
                "Attention: la verite qui ressort le plus clairement ici, c'est que la taille reste encore plus chargee que le reste de la silhouette. "
                "C'est cette zone qui freine le rendu global le plus net."
            )

    body_band = _classify_body_fat_band(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent)
    if (visual_features.definition_score or 0.0) < 0.48 and body_band in {"fit", "controlled", "moderate"}:
        return (
            "Attention: la definition visuelle est encore trop moyenne pour donner un rendu vraiment sec. "
            "Le physique parait propre, mais pas encore suffisamment tranche."
        )
    if body_band in {"fit", "controlled"}:
        return (
            "Attention: la base est deja bonne, mais il reste encore assez de body fat pour lisser une partie du rendu. "
            "On n'est pas encore sur un physique vraiment affute."
        )
    if body_band in {"moderate", "elevated"}:
        return (
            "Attention: le niveau de body fat reste encore suffisamment present pour masquer une partie claire de la definition. "
            "La marge de progression est encore reelle."
        )

    return "Attention: rien de majeur ne bloque la lecture, mais cette analyse reste une estimation visuelle et non une mesure medicale."


def _build_parfait_note(
    *,
    has_successful_detection: bool,
    overall_quality_score: float | None,
    confidence_score: float | None,
    analyzed_regions_summary: list[BodyRegionStatus],
    estimated_body_fat_percent: float | None,
    visual_features: AggregatedVisualFeatures,
    sex: str | None,
) -> str:
    if not has_successful_detection:
        return "Parfait: on n'a pas encore de base d'analyse assez propre pour valider un point fort."

    strong_regions = [region.label for region in analyzed_regions_summary if region.taken_into_account]
    if _is_upper_body_dominant_scan(analyzed_regions_summary=analyzed_regions_summary, visual_features=visual_features):
        return (
            f"Parfait: le haut du corps est deja bien capte et reste exploitable, surtout {_join_labels(_upper_body_strong_regions(analyzed_regions_summary)[:4])}. "
            "La lecture du torse et de la taille tient donc deja bien debout."
        )
    body_band = _classify_body_fat_band(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent)
    frame_signal = _has_frame_signal(sex=sex, visual_features=visual_features)
    definition_score = visual_features.definition_score or 0.0
    if (
        overall_quality_score is not None
        and overall_quality_score >= 0.75
        and confidence_score is not None
        and confidence_score >= 0.7
    ):
        region_fragment = ""
        if strong_regions:
            region_fragment = f" Les zones les plus propres sont {_join_labels(strong_regions[:4])}."
        if frame_signal and definition_score >= 0.64 and body_band in {"very_lean", "lean", "fit"}:
            return (
                f"Parfait: ce que le scan valide bien, c'est une base deja sportive, avec une structure visible et un rendu deja assez propre.{region_fragment}"
            )
        return (
            f"Parfait: ce qu'on peut valider proprement ici, c'est que le scan est net et que la lecture du physique tient bien debout.{region_fragment}"
        )

    if strong_regions:
        return (
            f"Parfait: plusieurs zones utiles sont bien lues par le scan, notamment {_join_labels(strong_regions[:3])}."
        )

    return "Parfait: la detection du corps fonctionne correctement, donc on a deja une base exploitable pour progresser."


def _build_progression_note(
    *,
    sex: str | None,
    confidence_score: float | None,
    estimated_body_fat_percent: float | None,
    estimated_lean_mass_kg: float | None,
    weight_kg: float | None,
    visual_features: AggregatedVisualFeatures,
    analyzed_regions_summary: list[BodyRegionStatus],
    reference_date: date | None,
) -> str:
    weak_regions = [region.label for region in analyzed_regions_summary if not region.taken_into_account]
    if confidence_score is not None and confidence_score < 0.45:
        return (
            "Progression: la priorite immediate n'est pas de tirer des conclusions sur le physique, mais d'obtenir un scan plus propre avec des photos plus nettes, plus completes et plus stables. "
        )
    if _is_upper_body_dominant_scan(analyzed_regions_summary=analyzed_regions_summary, visual_features=visual_features):
        return (
            "Progression: pour passer d'une bonne lecture du haut du corps a une analyse vraiment complete, ajoute un scan ou les hanches, cuisses et jambes sont visibles proprement. "
            "C'est ce qui manque le plus pour rendre le diagnostic plus solide."
        )
    if (visual_features.view_diversity_score or 0.0) < 0.45 and visual_features.usable_image_count <= 1:
        return (
            "Progression: pour faire monter la precision de l'analyse, ajoute au moins une vraie vue complementaire, idealement face plus profil dans les memes conditions. "
        )
    if weak_regions:
        return (
            f"Progression: avant d'aller chercher une analyse plus dure et plus juste, il faut d'abord fiabiliser {_join_labels(weak_regions[:3])} "
            "avec des photos face et profil mieux cadrees."
        )

    if _has_abdominal_attention_signal(
        sex=sex,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
    ):
        return (
            "Progression: le levier le plus rentable maintenant, c'est de continuer a faire descendre la zone taille / abdomen. "
            f"{_build_trajectory_note(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent, estimated_lean_mass_kg=estimated_lean_mass_kg, weight_kg=weight_kg, reference_date=reference_date)}"
        )

    coaching_feedback = build_coaching_feedback(
        sex=sex,
        confidence_score=confidence_score,
        estimated_body_fat_percent=estimated_body_fat_percent,
        estimated_lean_mass_kg=estimated_lean_mass_kg,
        weight_kg=weight_kg,
        visual_features=visual_features,
        reference_date=reference_date,
    )
    return f"Progression: {coaching_feedback}"


def _build_composition_note(
    *,
    sex: str | None,
    estimated_body_fat_percent: float,
    visual_features: AggregatedVisualFeatures,
) -> str:
    body_band = _classify_body_fat_band(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent)
    has_abdominal_signal = _has_abdominal_attention_signal(
        sex=sex,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
    )

    if body_band == "very_lean":
        return "Le rendu visuel montre deja un physique tres sec et deja avance."
    if body_band == "lean":
        return "Le rendu visuel montre deja une base athletique propre, avec peu de gras residuel."
    if body_band == "fit":
        if has_abdominal_signal:
            return "La base est bonne, mais le vrai residuel semble encore se jouer autour de la taille."
        if (visual_features.frame_score or 0.0) >= 0.64 and (visual_features.definition_score or 0.0) >= 0.6:
            return "La base visuelle parait deja athletique, avec une structure bien visible. La marge restante semble surtout se jouer sur les derniers details."
        return "La base est bonne et deja sportive, meme s'il reste encore du gras a perdre pour durcir vraiment le rendu."
    if body_band == "controlled":
        if has_abdominal_signal:
            return "La structure generale est correcte, mais le gras residuel semble encore surtout centre autour de la taille."
        return "La structure generale est correcte, mais le rendu reste encore trop lisse pour paraitre sec."
    if body_band == "moderate":
        return "Le potentiel de progression reste important, avec encore une marge nette pour affiner la silhouette."
    if body_band == "elevated":
        return "Le niveau de body fat estime reste assez eleve pour masquer une bonne partie de la definition."
    return "La composition generale parait exploitable au vu de l'estimation actuelle."


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
        return "Aucune projection cible ne peut etre proposee tant que le sexe n'est pas renseigne."
    if estimated_body_fat_percent <= target_body_fat:
        return (
            f"Tu sembles deja proche de la zone cible fixee autour de {target_body_fat:.0f}% de body fat. "
            "La progression la plus rentable devient surtout la regularite, la stabilisation et une meilleure definition visuelle."
        )

    target_weight = estimated_lean_mass_kg / (1.0 - (target_body_fat / 100.0))
    weight_to_lose = max(0.0, weight_kg - target_weight)
    if weight_to_lose <= 0.2:
        return (
            f"Tu sembles deja tres proche d'un objectif situe autour de {target_body_fat:.0f}% de body fat. "
            "La marge restante parait faible et devrait surtout se jouer sur les derniers details."
        )

    weekly_rate_kg = 0.5
    weeks_needed = weight_to_lose / weekly_rate_kg
    today = reference_date or date.today()
    projected_date = today + timedelta(days=ceil(weeks_needed * 7))
    month_label = _format_french_month(projected_date)
    return (
        f"Si le rythme reste regulier autour de {weekly_rate_kg:.1f} kg par semaine, "
        f"tu pourrais te rapprocher de {target_body_fat:.0f}% de body fat estime autour de {month_label}. "
        "L'idee n'est pas d'aller vite, mais de continuer a nettoyer progressivement la zone qui reste le plus en retrait."
    )


def _build_body_status_note(
    *,
    sex: str | None,
    estimated_body_fat_percent: float | None,
    visual_features: AggregatedVisualFeatures | None,
) -> str:
    if estimated_body_fat_percent is None:
        return "Une estimation complete du body fat n'a pas pu etre produite."

    body_band = _classify_body_fat_band(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent)
    has_abdominal_signal = _has_abdominal_attention_signal(
        sex=sex,
        estimated_body_fat_percent=estimated_body_fat_percent,
        visual_features=visual_features,
    )
    if body_band in {"very_lean", "lean"}:
        return "Le rendu visuel montre deja une base assez seche. L'estimation du body fat reste toutefois directionnelle, pas medicale."
    if has_abdominal_signal:
        return "Le rendu visuel montre une base correcte, avec un residuel qui semble surtout se concentrer autour de la taille ou de l'abdomen."
    if body_band in {"fit", "controlled"}:
        return "Le rendu visuel montre une base deja correcte, avec encore assez de marge avant un rendu vraiment plus sec."
    return "Le niveau de body fat estime reste encore assez present pour lisser une partie claire de la definition visuelle."


def _build_attention_title(
    *,
    has_successful_detection: bool,
    overall_quality_score: float | None,
    confidence_score: float | None,
    analyzed_regions_summary: list[BodyRegionStatus],
    estimated_body_fat_percent: float | None,
    visual_features: AggregatedVisualFeatures,
    sex: str | None,
) -> str:
    if not has_successful_detection:
        return "Scan a refaire"
    if _is_upper_body_dominant_scan(analyzed_regions_summary=analyzed_regions_summary, visual_features=visual_features):
        return "Bas du corps encore absent"
    if confidence_score is not None and confidence_score < 0.45:
        return "Confiance insuffisante"
    if overall_quality_score is not None and overall_quality_score < 0.6:
        return "Qualite photo a corriger"
    if (visual_features.view_diversity_score or 0.0) < 0.45 and visual_features.usable_image_count <= 1:
        return "Angles encore trop limites"
    weak_regions = [region.label for region in analyzed_regions_summary if not region.taken_into_account]
    if weak_regions:
        return "Scan encore incomplet"
    if _has_abdominal_attention_signal(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent, visual_features=visual_features):
        return "Taille encore chargee" if (sex or "").lower() == "female" else "Graisse abdominale encore visible"
    if (visual_features.definition_score or 0.0) < 0.48:
        return "Definition encore trop faible"
    body_band = _classify_body_fat_band(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent)
    if body_band in {"moderate", "elevated", "controlled"}:
        return "Body fat encore trop present"
    return "Point de vigilance"


def _build_parfait_title(
    *,
    has_successful_detection: bool,
    overall_quality_score: float | None,
    confidence_score: float | None,
    analyzed_regions_summary: list[BodyRegionStatus],
    estimated_body_fat_percent: float | None,
    visual_features: AggregatedVisualFeatures,
    sex: str | None,
) -> str:
    if not has_successful_detection:
        return "Flow operationnel"
    if _is_upper_body_dominant_scan(analyzed_regions_summary=analyzed_regions_summary, visual_features=visual_features):
        return "Haut du corps bien capte"
    strong_regions = [region.label for region in analyzed_regions_summary if region.taken_into_account]
    body_band = _classify_body_fat_band(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent)
    frame_signal = _has_frame_signal(sex=sex, visual_features=visual_features)
    definition_score = visual_features.definition_score or 0.0
    if body_band in {"very_lean", "lean"}:
        return "Base deja tres propre"
    if body_band == "fit" and frame_signal and definition_score >= 0.64:
        return "Base athletique credible"
    if frame_signal and definition_score >= 0.56:
        return "Structure solide"
    if (
        overall_quality_score is not None
        and overall_quality_score >= 0.75
        and confidence_score is not None
        and confidence_score >= 0.7
    ):
        return "Lecture fiable"
    if strong_regions:
        return "Points forts identifies"
    return "Base deja exploitable"


def _build_progression_title(
    *,
    sex: str | None,
    confidence_score: float | None,
    estimated_body_fat_percent: float | None,
    visual_features: AggregatedVisualFeatures,
    analyzed_regions_summary: list[BodyRegionStatus],
) -> str:
    weak_regions = [region.label for region in analyzed_regions_summary if not region.taken_into_account]
    if confidence_score is not None and confidence_score < 0.45:
        return "Priorite scan"
    if _is_upper_body_dominant_scan(analyzed_regions_summary=analyzed_regions_summary, visual_features=visual_features):
        return "Montrer le bas du corps"
    if (visual_features.view_diversity_score or 0.0) < 0.45 and visual_features.usable_image_count <= 1:
        return "Ajouter un vrai second angle"
    if weak_regions:
        return "Fiabiliser le scan"
    if _has_abdominal_attention_signal(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent, visual_features=visual_features):
        return "Objectif definition"
    if (visual_features.definition_score or 0.0) < 0.5:
        return "Durcir le rendu"
    body_band = _classify_body_fat_band(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent)
    if body_band in {"very_lean", "lean"}:
        return "Consolider l'avance"
    if body_band in {"fit", "controlled"}:
        return "Prochain levier concret"
    return "Levier principal"


def _target_body_fat(sex: str | None) -> float | None:
    canonical_sex = (sex or "").lower()
    if canonical_sex == "male":
        return 15.0
    if canonical_sex == "female":
        return 24.0
    return None


def _classify_body_fat_band(
    *,
    sex: str | None,
    estimated_body_fat_percent: float | None,
) -> str:
    if estimated_body_fat_percent is None:
        return "unknown"

    canonical_sex = (sex or "").lower()
    if canonical_sex == "male":
        if estimated_body_fat_percent <= 10.0:
            return "very_lean"
        if estimated_body_fat_percent <= 13.5:
            return "lean"
        if estimated_body_fat_percent <= 16.5:
            return "fit"
        if estimated_body_fat_percent <= 19.5:
            return "controlled"
        if estimated_body_fat_percent <= 24.0:
            return "moderate"
        return "elevated"

    if canonical_sex == "female":
        if estimated_body_fat_percent <= 18.0:
            return "very_lean"
        if estimated_body_fat_percent <= 22.0:
            return "lean"
        if estimated_body_fat_percent <= 26.0:
            return "fit"
        if estimated_body_fat_percent <= 30.0:
            return "controlled"
        if estimated_body_fat_percent <= 35.0:
            return "moderate"
        return "elevated"

    if estimated_body_fat_percent <= 16.0:
        return "lean"
    if estimated_body_fat_percent <= 24.0:
        return "fit"
    if estimated_body_fat_percent <= 30.0:
        return "controlled"
    return "moderate"


def _has_abdominal_attention_signal(
    *,
    sex: str | None,
    estimated_body_fat_percent: float | None,
    visual_features: AggregatedVisualFeatures | None,
) -> bool:
    if estimated_body_fat_percent is None or visual_features is None:
        return False

    waist_to_hip = visual_features.estimated_waist_to_hip_ratio
    waist_to_bbox = visual_features.estimated_waist_to_bbox_height_ratio
    canonical_sex = (sex or "").lower()
    if canonical_sex == "male":
        return bool(
            (waist_to_hip is not None and waist_to_hip >= 0.91)
            or (waist_to_bbox is not None and waist_to_bbox >= 0.20)
            or estimated_body_fat_percent >= 16.0
        )
    if canonical_sex == "female":
        return bool(
            (waist_to_hip is not None and waist_to_hip >= 0.81)
            or (waist_to_bbox is not None and waist_to_bbox >= 0.232)
            or estimated_body_fat_percent >= 25.0
        )
    return bool(estimated_body_fat_percent >= 20.0)


def _has_frame_signal(
    *,
    sex: str | None,
    visual_features: AggregatedVisualFeatures,
) -> bool:
    shoulder_to_hip = visual_features.estimated_shoulder_to_hip_ratio
    if shoulder_to_hip is None:
        return False
    canonical_sex = (sex or "").lower()
    if canonical_sex == "male":
        return shoulder_to_hip >= 1.12
    if canonical_sex == "female":
        return shoulder_to_hip >= 1.0
    return shoulder_to_hip >= 1.05


def _classify_reliability_level(
    *,
    has_successful_detection: bool,
    effective_confidence: float | None,
    overall_quality_score: float | None,
    weak_regions: list[str],
) -> str:
    if not has_successful_detection:
        return "insufficient"
    if effective_confidence is not None and effective_confidence >= 0.82 and (overall_quality_score or 0.0) >= 0.75 and not weak_regions:
        return "high"
    if effective_confidence is not None and effective_confidence >= 0.62:
        return "medium"
    return "limited"


def _classify_definition_level(
    *,
    sex: str | None,
    estimated_body_fat_percent: float | None,
    visual_features: AggregatedVisualFeatures,
) -> str:
    body_band = _classify_body_fat_band(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent)
    definition_score = visual_features.definition_score or 0.0
    if body_band in {"very_lean", "lean"} or definition_score >= 0.78:
        return "very_good"
    if body_band == "fit" or definition_score >= 0.62:
        return "moderate_to_good"
    if body_band == "controlled" or definition_score >= 0.46:
        return "average"
    return "soft"


def _classify_fat_distribution(
    *,
    sex: str | None,
    estimated_body_fat_percent: float | None,
    visual_features: AggregatedVisualFeatures,
) -> str:
    central_fat = visual_features.central_fat_score or 0.0
    if _has_abdominal_attention_signal(sex=sex, estimated_body_fat_percent=estimated_body_fat_percent, visual_features=visual_features):
        if central_fat >= 0.7:
            return "central_dominant"
        return "slightly_central"
    if central_fat <= 0.35:
        return "well_distributed"
    return "mixed_distribution"


def _classify_frame_assessment(
    *,
    sex: str | None,
    visual_features: AggregatedVisualFeatures,
) -> str:
    frame_score = visual_features.frame_score or 0.0
    if _has_frame_signal(sex=sex, visual_features=visual_features):
        if frame_score >= 0.72:
            return "strong_frame"
        return "balanced_frame"
    if frame_score >= 0.46:
        return "neutral_frame"
    return "compact_frame"


def _classify_scan_readiness(
    *,
    reliability_level: str,
    has_successful_detection: bool,
    weak_regions: list[str],
) -> str:
    if not has_successful_detection:
        return "retake_needed"
    if reliability_level == "high":
        return "ready_for_actionable_feedback"
    if reliability_level == "medium" and not weak_regions:
        return "usable_but_not_final"
    return "needs_better_scan"


def _classify_best_next_focus(
    *,
    reliability_level: str,
    definition_level: str,
    fat_distribution: str,
    weak_regions: list[str],
    view_coverage: str,
) -> str:
    if reliability_level == "limited" or weak_regions:
        return "improve_scan_quality"
    if view_coverage in {"single_angle_only", "narrow_multi_view"}:
        return "expand_view_coverage"
    if fat_distribution in {"central_dominant", "slightly_central"}:
        return "continue_improving_definition"
    if definition_level in {"average", "soft"}:
        return "reduce_remaining_body_fat"
    return "maintain_and_refine"


def _is_upper_body_dominant_scan(
    *,
    analyzed_regions_summary: list[BodyRegionStatus],
    visual_features: AggregatedVisualFeatures,
) -> bool:
    if (visual_features.body_coverage_score or 0.0) >= 0.58:
        return False

    region_map = {region.key: region for region in analyzed_regions_summary}
    upper_keys = {"head", "torso", "waist", "hips", "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm"}
    lower_keys = {"left_thigh", "right_thigh", "left_lower_leg", "right_lower_leg"}

    upper_visible = sum(1 for key in upper_keys if region_map.get(key) and region_map[key].taken_into_account)
    lower_visible = sum(1 for key in lower_keys if region_map.get(key) and region_map[key].taken_into_account)
    return upper_visible >= 3 and lower_visible <= 1


def _upper_body_strong_regions(analyzed_regions_summary: list[BodyRegionStatus]) -> list[str]:
    upper_keys = {"head", "torso", "waist", "hips", "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm"}
    return [
        region.label
        for region in analyzed_regions_summary
        if region.key in upper_keys and region.taken_into_account
    ]


def _build_truth_title(*, scan_profile: ScanProfile) -> str:
    if scan_profile.dominant_limitation == "central_fat":
        return "Le constat principal"
    if scan_profile.dominant_limitation == "lack_of_definition":
        return "Le constat physique"
    if scan_profile.dominant_limitation in {"scan_quality", "missing_view_coverage", "pose_stability"}:
        return "La verite du scan"
    return "Le point qui ressort"


def _build_truth_message(
    *,
    scan_profile: ScanProfile,
    confidence_score: float | None,
    estimated_body_fat_percent: float | None,
    sex: str | None,
) -> str:
    if scan_profile.dominant_limitation == "scan_quality":
        return "Pour l'instant, le scan ne permet pas encore un verdict aussi strict qu'il pourrait l'etre. La qualite brute des images limite encore la lecture."
    if scan_profile.dominant_limitation == "missing_view_coverage":
        return "Le scan repose encore sur des angles trop proches. Sans vraie vue complementaire, on ne peut pas pousser le diagnostic au maximum."
    if scan_profile.dominant_limitation == "pose_stability":
        return "La posture du scan reste trop peu neutre pour verrouiller une lecture premium. Le corps est vu, mais pas encore dans des conditions parfaites."
    if scan_profile.dominant_limitation == "central_fat":
        if (sex or "").lower() == "female":
            return "Le scan indique surtout que la zone taille reste encore celle qui freine le plus le rendu sec et net."
        return "Le scan indique surtout que le gras residuel reste encore le plus visible autour de l'abdomen."
    if scan_profile.dominant_limitation == "lack_of_definition":
        return "Le physique parait propre, mais encore trop lisse pour donner un rendu vraiment sec et tranche."
    if confidence_score is not None and confidence_score >= 0.8 and estimated_body_fat_percent is not None:
        return f"Le scan est assez solide pour dire que la lecture actuelle du physique tient bien debout autour de {estimated_body_fat_percent:.1f}% de body fat estime."
    return "Le scan donne une lecture utile du physique, mais avec encore une petite part d'incertitude normale pour une estimation visuelle."


def _build_strength_title(*, scan_profile: ScanProfile) -> str:
    if scan_profile.dominant_strength in {"scan_clarity", "multi_view_support"}:
        return "Point fort du scan"
    return "Point fort du physique"


def _build_strength_message(*, scan_profile: ScanProfile, analysis_notes: AnalysisNotes) -> str:
    if scan_profile.dominant_strength == "multi_view_support":
        return "Le vrai plus ici, c'est que les angles fournis renforcent la solidite de l'analyse. On lit mieux le physique quand la vue de face et une vue complementaire se confirment."
    if scan_profile.dominant_strength == "scan_clarity":
        return "Le scan est suffisamment propre pour soutenir un retour plus dur et plus utile. La qualite de lecture est deja un vrai point fort."
    if scan_profile.dominant_strength == "visible_definition":
        return "Le rendu montre deja de la definition visible. C'est ce qui donne le plus de credibilite a l'impression de physique propre."
    if scan_profile.dominant_strength == "upper_body_structure":
        return "La structure du haut du corps ressort bien. C'est ce qui porte le mieux la lecture athletique du scan."
    if scan_profile.dominant_strength == "usable_body_read":
        return "Le scan capte deja plusieurs zones de maniere exploitable, ce qui donne une base serieuse a l'analyse."
    return analysis_notes.parfait.message


def _build_limitation_title(*, scan_profile: ScanProfile) -> str:
    if scan_profile.dominant_limitation == "central_fat":
        return "Frein principal"
    if scan_profile.dominant_limitation in {"scan_quality", "missing_view_coverage", "pose_stability"}:
        return "Limite du scan"
    return "Limite principale"


def _build_limitation_message(*, scan_profile: ScanProfile, analysis_notes: AnalysisNotes) -> str:
    if scan_profile.dominant_limitation == "scan_quality":
        return "La qualite des images reste encore la premiere chose qui empeche l'analyse de monter d'un cran."
    if scan_profile.dominant_limitation == "missing_view_coverage":
        return "Le manque d'angles vraiment complementaires reste aujourd'hui la principale limite de precision."
    if scan_profile.dominant_limitation == "pose_stability":
        return "La posture n'est pas encore assez neutre pour stabiliser totalement la lecture morphologique."
    if scan_profile.dominant_limitation == "central_fat":
        return "La retention autour de la taille ou de l'abdomen reste ce qui freine le plus un rendu sec et plus marque."
    if scan_profile.dominant_limitation == "lack_of_definition":
        return "Le manque de definition reste aujourd'hui la limite la plus visible du rendu global."
    return analysis_notes.attention.message


def _build_next_focus_title(*, scan_profile: ScanProfile) -> str:
    if scan_profile.best_next_focus == "expand_view_coverage":
        return "Priorite scan"
    if scan_profile.best_next_focus == "improve_scan_quality":
        return "Priorite technique"
    if scan_profile.best_next_focus == "continue_improving_definition":
        return "Priorite physique"
    return "Prochain focus"


def _build_next_focus_message(*, scan_profile: ScanProfile, analysis_notes: AnalysisNotes) -> str:
    if scan_profile.best_next_focus == "expand_view_coverage":
        return "Ajoute une vraie vue complementaire, idealement face plus profil, pour faire monter franchement la precision de lecture."
    if scan_profile.best_next_focus == "improve_scan_quality":
        return "Le meilleur gain immediate vient encore de photos plus propres, plus stables et mieux cadrees."
    if scan_profile.best_next_focus == "continue_improving_definition":
        return "Le levier le plus rentable maintenant reste de faire descendre la zone qui garde encore le plus de gras residuel."
    if scan_profile.best_next_focus == "reduce_remaining_body_fat":
        return "Le plus utile maintenant est de continuer a reduire le body fat residuel pour faire monter la definition visible."
    return analysis_notes.progression.message


def _build_scan_quality_message(
    *,
    scan_profile: ScanProfile,
    confidence_score: float | None,
    overall_quality_score: float | None,
) -> str:
    del confidence_score, overall_quality_score
    return (
        f"{scan_profile.confidence_message} "
        f"La couverture actuelle est {scan_profile.view_coverage} et la posture est evaluee comme {scan_profile.pose_quality}."
    )


def _classify_confidence_label(
    *,
    reliability_level: str,
    view_coverage: str,
    pose_quality: str,
) -> str:
    if reliability_level == "high" and view_coverage == "front_and_side_available" and pose_quality == "neutral":
        return "Tres fiable"
    if reliability_level == "high":
        return "Fiable"
    if reliability_level == "medium":
        return "A confirmer"
    return "Limitee"


def _build_confidence_message(
    *,
    reliability_level: str,
    view_coverage: str,
    pose_quality: str,
    weak_regions: list[str],
) -> str:
    if reliability_level == "insufficient":
        return "Le scan n'est pas encore assez solide pour soutenir une lecture corporelle fiable."
    if reliability_level == "limited":
        if weak_regions:
            return (
                f"La lecture reste limitee, car plusieurs zones du corps sont encore trop fragiles, surtout {_join_labels(weak_regions[:3])}."
            )
        return "La lecture reste limitee, car le scan reste encore trop partiel pour porter un retour vraiment dur."
    if reliability_level == "medium":
        if view_coverage in {"single_angle_only", "narrow_multi_view"}:
            return "Le scan est exploitable, mais il manque encore une vraie vue complementaire pour verrouiller le diagnostic."
        if pose_quality == "non_neutral":
            return "Le scan est exploitable, mais la posture reste encore trop variable pour une lecture premium."
        if weak_regions:
            return (
                f"Le scan est utilisable, mais certaines zones demandent encore confirmation, surtout {_join_labels(weak_regions[:3])}."
            )
        return "Le scan est deja utile, mais il reste encore un peu de marge avant une lecture pleinement verrouillee."
    if view_coverage == "front_and_side_available" and pose_quality == "neutral" and not weak_regions:
        return "Le scan est tres fiable: les angles sont bons, la posture est stable et la lecture globale tient bien."
    if weak_regions:
        return (
            f"Le scan est fiable dans l'ensemble, meme si quelques zones restent plus fragiles, surtout {_join_labels(weak_regions[:3])}."
        )
    return "Le scan est fiable et suffisamment propre pour soutenir un retour direct sur le physique."


def _classify_dominant_strength(
    *,
    reliability_level: str,
    definition_level: str,
    frame_assessment: str,
    view_coverage: str,
    analyzed_regions_summary: list[BodyRegionStatus],
) -> str:
    strong_region_count = sum(1 for region in analyzed_regions_summary if region.taken_into_account)
    if view_coverage == "front_and_side_available":
        return "multi_view_support"
    if reliability_level == "high" and strong_region_count >= 6:
        return "scan_clarity"
    if definition_level in {"very_good", "moderate_to_good"}:
        return "visible_definition"
    if frame_assessment in {"strong_frame", "balanced_frame"}:
        return "upper_body_structure"
    if strong_region_count >= 4:
        return "usable_body_read"
    return "basic_detection"


def _classify_dominant_limitation(
    *,
    reliability_level: str,
    fat_distribution: str,
    definition_level: str,
    weak_regions: list[str],
    view_coverage: str,
    pose_quality: str,
) -> str:
    if reliability_level in {"limited", "insufficient"} or weak_regions:
        return "scan_quality"
    if view_coverage in {"single_angle_only", "narrow_multi_view"}:
        return "missing_view_coverage"
    if pose_quality == "non_neutral":
        return "pose_stability"
    if fat_distribution in {"central_dominant", "slightly_central"}:
        return "central_fat"
    if definition_level in {"average", "soft"}:
        return "lack_of_definition"
    return "minor_residual_gap"


def _classify_view_coverage(*, visual_features: AggregatedVisualFeatures) -> str:
    if visual_features.front_view_count >= 1 and visual_features.side_view_count >= 1:
        return "front_and_side_available"
    if visual_features.view_diversity_score is not None and visual_features.view_diversity_score >= 0.75:
        return "multi_angle_supported"
    if visual_features.usable_image_count >= 2:
        return "narrow_multi_view"
    return "single_angle_only"


def _classify_pose_quality(*, visual_features: AggregatedVisualFeatures) -> str:
    score = visual_features.pose_neutrality_score or 0.0
    if score >= 0.8:
        return "neutral"
    if score >= 0.58:
        return "mostly_neutral"
    return "non_neutral"


def _build_scan_profile_summary(
    *,
    reliability_level: str,
    definition_level: str,
    fat_distribution: str,
    frame_assessment: str,
    view_coverage: str,
    pose_quality: str,
    best_next_focus: str,
    dominant_strength: str,
    dominant_limitation: str,
) -> str:
    reliability_note = {
        "high": "Le scan est suffisamment propre pour une lecture exploitable.",
        "medium": "Le scan est deja exploitable, mais pas encore au maximum de sa precision.",
        "limited": "Le scan donne une tendance utile, mais la fiabilite reste encore moyenne.",
        "insufficient": "Le scan n'est pas encore assez solide pour une lecture fiable.",
    }[reliability_level]

    definition_note = {
        "very_good": "La definition visuelle parait deja bien marquee.",
        "moderate_to_good": "La definition visuelle parait deja correcte.",
        "average": "La definition parait intermediaire, sans etre encore tres marquee.",
        "soft": "La definition visuelle parait encore assez lisse.",
    }[definition_level]

    distribution_note = {
        "central_dominant": "La marge principale semble surtout se concentrer autour de la taille.",
        "slightly_central": "La marge la plus visible semble legerement centree autour de la taille.",
        "well_distributed": "La repartition visuelle parait assez harmonieuse.",
        "mixed_distribution": "La repartition visuelle parait melangee sans point unique trop dominant.",
    }[fat_distribution]

    frame_note = {
        "strong_frame": "La structure du haut du corps parait bien presente.",
        "balanced_frame": "La structure generale parait equilibree.",
        "neutral_frame": "La structure du physique parait plutot neutre.",
        "compact_frame": "La structure visuelle reste assez compacte.",
    }[frame_assessment]

    view_note = {
        "front_and_side_available": "Les angles fournis couvrent bien le scan.",
        "multi_angle_supported": "Le scan profite deja de plusieurs angles utiles.",
        "narrow_multi_view": "Le scan a plusieurs images, mais les angles restent encore proches.",
        "single_angle_only": "Le scan repose encore surtout sur un seul angle.",
    }[view_coverage]

    pose_note = {
        "neutral": "La posture parait suffisamment neutre pour stabiliser la lecture.",
        "mostly_neutral": "La posture reste globalement exploitable.",
        "non_neutral": "La posture reste encore trop variable pour une lecture maximale.",
    }[pose_quality]

    focus_note = {
        "improve_scan_quality": "Le meilleur levier immediat reste d'ameliorer le scan.",
        "expand_view_coverage": "Le meilleur levier immediat est d'ajouter une vraie vue complementaire.",
        "continue_improving_definition": "Le meilleur levier semble etre de continuer a gagner en definition.",
        "reduce_remaining_body_fat": "Le meilleur levier reste de continuer a reduire le body fat residuel.",
        "maintain_and_refine": "Le meilleur levier devient surtout la regularite et les derniers details.",
    }[best_next_focus]

    strength_note = {
        "multi_view_support": "Le fait d'avoir plusieurs angles utiles renforce clairement la lecture du scan.",
        "scan_clarity": "Le scan est suffisamment clair pour appuyer un retour plus direct.",
        "visible_definition": "La definition visuelle est deja un vrai point fort du rendu.",
        "upper_body_structure": "La structure du haut du corps ressort comme un vrai point fort.",
        "usable_body_read": "Plusieurs zones du corps sont deja bien lues par le scan.",
        "basic_detection": "La detection du corps est bien en place.",
    }[dominant_strength]

    limitation_note = {
        "scan_quality": "La limite principale reste encore la qualite ou la couverture du scan.",
        "missing_view_coverage": "La limite principale reste le manque d'angles vraiment complementaires.",
        "pose_stability": "La limite principale reste une posture encore trop peu neutre.",
        "central_fat": "La limite principale semble rester une retention plus visible autour de la taille.",
        "lack_of_definition": "La limite principale reste un manque de definition visuelle.",
        "minor_residual_gap": "La limite principale parait residuelle et plutot moderee.",
    }[dominant_limitation]

    return f"{reliability_note} {view_note} {pose_note} {strength_note} {frame_note} {definition_note} {distribution_note} {limitation_note} {focus_note}".strip()


def _format_french_month(value: date) -> str:
    month_names = {
        1: "janvier",
        2: "fevrier",
        3: "mars",
        4: "avril",
        5: "mai",
        6: "juin",
        7: "juillet",
        8: "aout",
        9: "septembre",
        10: "octobre",
        11: "novembre",
        12: "decembre",
    }
    return f"{month_names[value.month]} {value.year}"
