"""Image upload validation and decoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile


SUPPORTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class UnsupportedImageFormatError(ValueError):
    """Raised when the uploaded file type is not supported."""


class ImageLoadError(ValueError):
    """Raised when the image bytes cannot be decoded by OpenCV."""


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """Decoded image and associated metadata."""

    filename: str
    content_type: str | None
    image_bgr: np.ndarray
    width: int
    height: int


def validate_upload_format(upload_file: UploadFile) -> None:
    """Validate file extension and content type before decoding."""

    filename = upload_file.filename or "uploaded_image"
    extension = Path(filename).suffix.lower()
    content_type = (upload_file.content_type or "").lower()

    if extension and extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedImageFormatError(
            f"Unsupported file extension '{extension}'. Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}."
        )
    if content_type and content_type not in SUPPORTED_CONTENT_TYPES:
        raise UnsupportedImageFormatError(
            f"Unsupported content type '{content_type}'. Supported types: {sorted(SUPPORTED_CONTENT_TYPES)}."
        )


async def load_image_from_upload(upload_file: UploadFile, max_file_size_mb: int) -> LoadedImage:
    """Read and decode an uploaded image with OpenCV."""

    validate_upload_format(upload_file)
    raw_bytes = await upload_file.read()
    if not raw_bytes:
        raise ImageLoadError("Uploaded image is empty.")

    max_bytes = max_file_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise ImageLoadError(f"Uploaded image exceeds the {max_file_size_mb} MB limit.")

    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ImageLoadError("Uploaded file could not be decoded as an image.")

    height, width = image_bgr.shape[:2]
    return LoadedImage(
        filename=upload_file.filename or "uploaded_image",
        content_type=upload_file.content_type,
        image_bgr=image_bgr,
        width=width,
        height=height,
    )
