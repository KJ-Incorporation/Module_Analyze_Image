"""FastAPI entrypoint for the Weighty vision module."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as analysis_router
from app.core.config import get_settings
from app.services.pose_estimator import MediaPipePoseEstimator

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services at startup."""

    settings = get_settings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    app.state.settings = settings
    app.state.pose_estimator = MediaPipePoseEstimator(settings)
    if not app.state.pose_estimator.is_available:
        logger.warning(
            "Pose estimator initialized in unavailable state: %s",
            app.state.pose_estimator.initialization_error,
        )
    else:
        logger.info("Pose estimator initialized successfully.")

    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(analysis_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Lightweight health check."""

    return {"status": "ok"}


@app.get("/test-ui", response_class=FileResponse)
async def test_ui() -> FileResponse:
    """Serve the lightweight HTML test UI from the FastAPI backend."""

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/test-food-ui", response_class=FileResponse)
async def test_food_ui() -> FileResponse:
    """Serve the lightweight HTML test UI for food analysis."""

    return FileResponse(STATIC_DIR / "food-test-ui.html")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return JSON for unexpected errors."""

    logging.getLogger(__name__).exception("Unhandled error while serving %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )
