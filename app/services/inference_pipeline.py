"""End-to-end inference pipeline for body fat estimation."""

from __future__ import annotations

import logging

from fastapi import UploadFile

from app.core.config import Settings
from app.schemas.request import AnalyzeRequestMetadata
from app.schemas.response import (
    AnalysisBlocksResponse,
    AnalysisNotesBlockResponse,
    AnalysisNotesResponse,
    AnalyzeResponse,
    BodyRegionStatusResponse,
    BoundingBoxResponse,
    ImageAnalysisResponse,
    LandmarkResponse,
    ScanProfileResponse,
    SomatotypeResponse,
    SomatotypeScoresResponse,
)
from app.services.body_metrics import (
    aggregate_region_summaries,
    derive_body_metrics,
    summarize_analyzed_regions,
)
from app.services.bodyfat_estimator import estimate_body_fat, estimate_somatotype, normalize_sex
from app.services.feature_engineering import aggregate_visual_features, build_image_feature_set
from app.services.image_loader import (
    ImageLoadError,
    UnsupportedImageFormatError,
    load_image_from_upload,
)
from app.services.pose_estimator import (
    MediaPipePoseEstimator,
    PoseEstimationError,
    PoseEstimatorInitializationError,
)
from app.services.quality_checks import (
    assess_image_quality,
    build_analysis_feedback,
    build_analysis_blocks,
    build_analysis_notes,
    build_coaching_feedback,
    build_recommendations,
    build_scan_profile,
    calculate_overall_quality_score,
)

LOGGER = logging.getLogger(__name__)


class BodyFatInferencePipeline:
    """Application service that orchestrates multi-image inference."""

    def __init__(self, settings: Settings, pose_estimator: MediaPipePoseEstimator) -> None:
        self._settings = settings
        self._pose_estimator = pose_estimator

    async def analyze(
        self,
        *,
        metadata: AnalyzeRequestMetadata,
        images: list[UploadFile],
    ) -> AnalyzeResponse:
        """Analyze images and return the consolidated response payload."""

        results: list[ImageAnalysisResponse] = []
        feature_sets = []
        quality_scores: list[float] = []
        analyzed_region_sets = []
        has_successful_detection = False
        has_blurry_image = False
        has_invalid_image = False
        has_missing_torso_metrics = False

        for upload in images:
            LOGGER.info("Analyzing uploaded image '%s'", upload.filename)
            try:
                loaded_image = await load_image_from_upload(upload, self._settings.max_file_size_mb)
            except UnsupportedImageFormatError as exc:
                has_invalid_image = True
                results.append(
                    _build_failed_image_response(
                        upload=upload,
                        processing_status="unsupported_format",
                        warning=str(exc),
                    )
                )
                continue
            except ImageLoadError as exc:
                has_invalid_image = True
                results.append(
                    _build_failed_image_response(
                        upload=upload,
                        processing_status="invalid_image",
                        warning=str(exc),
                    )
                )
                continue

            quality = assess_image_quality(loaded_image.image_bgr, self._settings)
            quality_scores.append(quality.quality_score)
            has_blurry_image = has_blurry_image or quality.is_blurry

            try:
                pose_result = self._pose_estimator.estimate_pose(loaded_image.image_bgr)
            except PoseEstimatorInitializationError:
                raise
            except PoseEstimationError as exc:
                LOGGER.exception("Pose estimation failed for '%s'", upload.filename)
                has_invalid_image = True
                results.append(
                    _build_failed_image_response(
                        upload=upload,
                        processing_status="processing_error",
                        warning=str(exc),
                        image_width=loaded_image.width,
                        image_height=loaded_image.height,
                        quality_score=quality.quality_score,
                    )
                )
                continue

            warnings = [*quality.warnings, *pose_result.warnings]
            if not pose_result.body_detected:
                results.append(
                    ImageAnalysisResponse(
                        filename=loaded_image.filename,
                        content_type=loaded_image.content_type,
                        processing_status="no_body_detected",
                        body_detected=False,
                        confidence_score=pose_result.confidence_score,
                        quality_score=quality.quality_score,
                        usable_for_body_fat_estimation=False,
                        image_width=loaded_image.width,
                        image_height=loaded_image.height,
                        bbox=None,
                        landmarks=[],
                        analyzed_regions=[],
                        estimated_shoulder_width_px=None,
                        estimated_hip_width_px=None,
                        estimated_waist_width_px=None,
                        estimated_waist_to_hip_ratio=None,
                        posture_summary=None,
                        warnings=_deduplicate_strings(warnings),
                    )
                )
                has_missing_torso_metrics = True
                continue

            has_successful_detection = True
            derived_metrics = derive_body_metrics(
                pose_result.landmarks,
                visibility_threshold=self._settings.metric_visibility_threshold,
            )
            warnings.extend(derived_metrics.warnings)
            if (
                derived_metrics.estimated_shoulder_width_px is None
                or derived_metrics.estimated_hip_width_px is None
                or derived_metrics.estimated_waist_width_px is None
            ):
                has_missing_torso_metrics = True

            bbox_response = (
                BoundingBoxResponse(
                    x_min=pose_result.bbox.x_min,
                    y_min=pose_result.bbox.y_min,
                    x_max=pose_result.bbox.x_max,
                    y_max=pose_result.bbox.y_max,
                    width=pose_result.bbox.width,
                    height=pose_result.bbox.height,
                )
                if pose_result.bbox is not None
                else None
            )
            feature_set = build_image_feature_set(
                body_metrics=derived_metrics,
                bbox=bbox_response,
                quality_score=quality.quality_score,
                pose_confidence_score=pose_result.confidence_score,
                posture_summary=derived_metrics.posture_summary,
            )
            feature_sets.append(feature_set)
            analyzed_regions = summarize_analyzed_regions(
                pose_result.landmarks,
                visibility_threshold=self._settings.metric_visibility_threshold,
                image_width=loaded_image.width,
                image_height=loaded_image.height,
            )
            analyzed_region_sets.append(analyzed_regions)

            results.append(
                ImageAnalysisResponse(
                    filename=loaded_image.filename,
                    content_type=loaded_image.content_type,
                    processing_status="success",
                    body_detected=True,
                    confidence_score=pose_result.confidence_score,
                    quality_score=quality.quality_score,
                    usable_for_body_fat_estimation=feature_set.usable_for_body_fat_estimation,
                    image_width=loaded_image.width,
                    image_height=loaded_image.height,
                    bbox=bbox_response,
                    landmarks=[
                        LandmarkResponse(
                            index=landmark.index,
                            name=landmark.name,
                            x_px=landmark.x_px,
                            y_px=landmark.y_px,
                            visibility=landmark.visibility,
                            presence=landmark.presence,
                        )
                        for landmark in pose_result.landmarks
                    ],
                    analyzed_regions=[
                        BodyRegionStatusResponse(
                            key=region.key,
                            label=region.label,
                            visible=region.visible,
                            confidence=region.confidence,
                            taken_into_account=region.taken_into_account,
                        )
                        for region in analyzed_regions
                    ],
                    estimated_shoulder_width_px=derived_metrics.estimated_shoulder_width_px,
                    estimated_hip_width_px=derived_metrics.estimated_hip_width_px,
                    estimated_waist_width_px=derived_metrics.estimated_waist_width_px,
                    estimated_waist_to_hip_ratio=derived_metrics.estimated_waist_to_hip_ratio,
                    posture_summary=derived_metrics.posture_summary,
                    warnings=_deduplicate_strings([*warnings, *feature_set.warnings]),
                )
            )

        aggregated_visual_features = aggregate_visual_features(feature_sets)
        normalized_sex = normalize_sex(metadata.resolved_sex)
        estimate = estimate_body_fat(
            age=metadata.age,
            sex=normalized_sex,
            height_cm=metadata.height_cm,
            weight_kg=metadata.weight_kg,
            visual_features=aggregated_visual_features,
            model_version=self._settings.body_fat_model_version,
        )
        somatotype = estimate_somatotype(
            sex=normalized_sex or metadata.resolved_sex,
            bmi=estimate.bmi,
            visual_features=aggregated_visual_features,
            body_fat_percent=estimate.estimated_body_fat_percent,
        )
        overall_quality_score = calculate_overall_quality_score(quality_scores)
        recommendations = build_recommendations(
            has_successful_detection=has_successful_detection,
            has_blurry_image=has_blurry_image,
            has_invalid_image=has_invalid_image,
            missing_bmi_inputs=(metadata.height_cm is None or metadata.weight_kg is None),
            overall_quality_score=overall_quality_score,
            has_missing_torso_metrics=has_missing_torso_metrics,
            missing_body_fat_inputs=(
                metadata.age is None
                or normalized_sex is None
                or metadata.height_cm is None
                or metadata.weight_kg is None
            ),
            low_body_fat_confidence=(
                estimate.confidence_score is not None and estimate.confidence_score < 0.45
            ),
        )

        response_warnings = _deduplicate_strings(
            [
                *estimate.warnings,
                *[warning for result in results for warning in result.warnings],
            ]
        )
        analyzed_regions_summary = aggregate_region_summaries(analyzed_region_sets)
        analysis_feedback = build_analysis_feedback(
            has_successful_detection=has_successful_detection,
            overall_quality_score=overall_quality_score,
            confidence_score=estimate.confidence_score,
            estimated_body_fat_percent=estimate.estimated_body_fat_percent,
            analyzed_regions_summary=analyzed_regions_summary,
            sex=normalized_sex or metadata.resolved_sex,
            visual_features=aggregated_visual_features,
        )
        coaching_feedback = build_coaching_feedback(
            sex=normalized_sex or metadata.resolved_sex,
            confidence_score=estimate.confidence_score,
            estimated_body_fat_percent=estimate.estimated_body_fat_percent,
            estimated_lean_mass_kg=estimate.estimated_lean_mass_kg,
            weight_kg=metadata.weight_kg,
            visual_features=aggregated_visual_features,
        )
        analysis_notes = build_analysis_notes(
            has_successful_detection=has_successful_detection,
            overall_quality_score=overall_quality_score,
            confidence_score=estimate.confidence_score,
            estimated_body_fat_percent=estimate.estimated_body_fat_percent,
            estimated_lean_mass_kg=estimate.estimated_lean_mass_kg,
            weight_kg=metadata.weight_kg,
            sex=normalized_sex or metadata.resolved_sex,
            analyzed_regions_summary=analyzed_regions_summary,
            visual_features=aggregated_visual_features,
        )
        scan_profile = build_scan_profile(
            has_successful_detection=has_successful_detection,
            confidence_score=estimate.confidence_score,
            overall_quality_score=overall_quality_score,
            estimated_body_fat_percent=estimate.estimated_body_fat_percent,
            analyzed_regions_summary=analyzed_regions_summary,
            visual_features=aggregated_visual_features,
            sex=normalized_sex or metadata.resolved_sex,
        )
        analysis_blocks = build_analysis_blocks(
            analysis_notes=analysis_notes,
            scan_profile=scan_profile,
            confidence_score=estimate.confidence_score,
            overall_quality_score=overall_quality_score,
            estimated_body_fat_percent=estimate.estimated_body_fat_percent,
            sex=normalized_sex or metadata.resolved_sex,
        )
        return AnalyzeResponse(
            user_id=metadata.user_id,
            sex=normalized_sex or metadata.resolved_sex,
            age=metadata.age,
            height_cm=metadata.height_cm,
            weight_kg=metadata.weight_kg,
            images_processed=len(results),
            bmi=estimate.bmi,
            estimated_body_fat_percent=estimate.estimated_body_fat_percent,
            estimated_fat_mass_kg=estimate.estimated_fat_mass_kg,
            estimated_lean_mass_kg=estimate.estimated_lean_mass_kg,
            confidence_score=estimate.confidence_score,
            model_version=estimate.model_version,
            estimated_body_fat_note=(
                "Image-based body fat is an estimate only and must not be treated as a medical measurement."
            ),
            analysis_feedback=analysis_feedback,
            coaching_feedback=coaching_feedback,
            analysis_notes=AnalysisNotesBlockResponse(
                attention=AnalysisNotesResponse(
                    title=analysis_notes.attention.title,
                    message=analysis_notes.attention.message,
                ),
                parfait=AnalysisNotesResponse(
                    title=analysis_notes.parfait.title,
                    message=analysis_notes.parfait.message,
                ),
                progression=AnalysisNotesResponse(
                    title=analysis_notes.progression.title,
                    message=analysis_notes.progression.message,
                ),
            ),
            analysis_blocks=AnalysisBlocksResponse(
                overview=AnalysisNotesResponse(
                    title=analysis_blocks.overview.title,
                    message=analysis_blocks.overview.message,
                ),
                truth=AnalysisNotesResponse(
                    title=analysis_blocks.truth.title,
                    message=analysis_blocks.truth.message,
                ),
                strength=AnalysisNotesResponse(
                    title=analysis_blocks.strength.title,
                    message=analysis_blocks.strength.message,
                ),
                limitation=AnalysisNotesResponse(
                    title=analysis_blocks.limitation.title,
                    message=analysis_blocks.limitation.message,
                ),
                next_focus=AnalysisNotesResponse(
                    title=analysis_blocks.next_focus.title,
                    message=analysis_blocks.next_focus.message,
                ),
                scan_quality=AnalysisNotesResponse(
                    title=analysis_blocks.scan_quality.title,
                    message=analysis_blocks.scan_quality.message,
                ),
            ),
            scan_profile=ScanProfileResponse(
                reliability_level=scan_profile.reliability_level,
                confidence_label=scan_profile.confidence_label,
                confidence_message=scan_profile.confidence_message,
                definition_level=scan_profile.definition_level,
                fat_distribution=scan_profile.fat_distribution,
                frame_assessment=scan_profile.frame_assessment,
                view_coverage=scan_profile.view_coverage,
                pose_quality=scan_profile.pose_quality,
                scan_readiness=scan_profile.scan_readiness,
                best_next_focus=scan_profile.best_next_focus,
                dominant_strength=scan_profile.dominant_strength,
                dominant_limitation=scan_profile.dominant_limitation,
                summary=scan_profile.summary,
            ),
            somatotype=SomatotypeResponse(
                primary=somatotype.primary,
                secondary=somatotype.secondary,
                confidence=somatotype.confidence,
                scores=SomatotypeScoresResponse(
                    ectomorph=somatotype.ectomorph_score,
                    mesomorph=somatotype.mesomorph_score,
                    endomorph=somatotype.endomorph_score,
                ),
                notes=somatotype.notes,
            ),
            overall_quality_score=overall_quality_score,
            warnings=response_warnings,
            recommendations=recommendations,
            analyzed_regions_summary=[
                BodyRegionStatusResponse(
                    key=region.key,
                    label=region.label,
                    visible=region.visible,
                    confidence=region.confidence,
                    taken_into_account=region.taken_into_account,
                )
                for region in analyzed_regions_summary
            ],
            results=results,
        )


def _build_failed_image_response(
    *,
    upload: UploadFile,
    processing_status: str,
    warning: str,
    image_width: int | None = None,
    image_height: int | None = None,
    quality_score: float | None = None,
) -> ImageAnalysisResponse:
    return ImageAnalysisResponse(
        filename=upload.filename or "uploaded_image",
        content_type=upload.content_type,
        processing_status=processing_status,
        body_detected=False,
        confidence_score=None,
        quality_score=quality_score,
        usable_for_body_fat_estimation=False,
        image_width=image_width,
        image_height=image_height,
        bbox=None,
        landmarks=[],
        analyzed_regions=[],
        estimated_shoulder_width_px=None,
        estimated_hip_width_px=None,
        estimated_waist_width_px=None,
        estimated_waist_to_hip_ratio=None,
        posture_summary=None,
        warnings=[warning],
    )


def _deduplicate_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
