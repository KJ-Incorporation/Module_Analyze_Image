"""API routes for the Weighty vision module."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.schemas.request import AnalyzeRequestMetadata
from app.schemas.response import AnalyzeResponse
from app.services.inference_pipeline import BodyFatInferencePipeline
from app.services.pose_estimator import PoseEstimatorInitializationError

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze one or more human photos",
)
async def analyze_images(
    request: Request,
    metadata: Annotated[AnalyzeRequestMetadata, Depends(AnalyzeRequestMetadata.as_form)],
    images: Annotated[list[UploadFile], File(description="One or more JPEG, PNG, or WEBP files.")],
) -> AnalyzeResponse:
    """Analyze uploaded images and return JSON-only body fat estimates."""

    settings = request.app.state.settings
    pipeline = BodyFatInferencePipeline(
        settings=settings,
        pose_estimator=request.app.state.pose_estimator,
    )

    if not images:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one image must be provided.",
        )
    if len(images) > settings.max_images_per_request:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"A maximum of {settings.max_images_per_request} images is allowed per request.",
        )
    try:
        return await pipeline.analyze(metadata=metadata, images=images)
    except PoseEstimatorInitializationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
