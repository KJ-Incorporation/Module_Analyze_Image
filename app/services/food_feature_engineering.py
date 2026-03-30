"""Visual feature extraction for food analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
else:
    np = Any


@dataclass(frozen=True, slots=True)
class FoodVisualFeatures:
    """Lightweight food-oriented image features."""

    mean_brightness: float
    mean_saturation: float
    edge_density: float
    colorfulness_score: float
    food_area_score: float
    green_share: float
    red_share: float
    yellow_share: float
    brown_share: float
    white_share: float
    orange_share: float


def extract_food_visual_features(image_bgr: np.ndarray) -> FoodVisualFeatures:
    """Extract simple color and texture cues from a decoded food image."""

    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    mean_brightness = float(gray.mean()) / 255.0
    mean_saturation = float(hsv[:, :, 1].mean()) / 255.0
    edge_density = float((edges > 0).mean())

    b_channel, g_channel, r_channel = cv2.split(image_bgr.astype("float32"))
    rg = np.abs(r_channel - g_channel)
    yb = np.abs(0.5 * (r_channel + g_channel) - b_channel)
    colorfulness_score = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) / 128.0)
    colorfulness_score = max(0.0, min(1.0, colorfulness_score))

    food_area_mask = ((hsv[:, :, 1] > 28) | (gray < 225)).astype("uint8")
    food_area_score = float(food_area_mask.mean())

    green_share = _mask_share(hsv, ((35, 85), (40, 255), (35, 255)))
    yellow_share = _mask_share(hsv, ((15, 35), (40, 255), (80, 255)))
    orange_share = _mask_share(hsv, ((6, 20), (70, 255), (60, 255)))
    brown_share = _mask_share(hsv, ((5, 25), (45, 255), (20, 190)))
    white_share = _mask_share(hsv, ((0, 180), (0, 45), (180, 255)))
    red_share = _mask_share(hsv, ((0, 10), (60, 255), (45, 255))) + _mask_share(
        hsv, ((170, 180), (60, 255), (45, 255))
    )

    return FoodVisualFeatures(
        mean_brightness=round(mean_brightness, 3),
        mean_saturation=round(mean_saturation, 3),
        edge_density=round(edge_density, 3),
        colorfulness_score=round(colorfulness_score, 3),
        food_area_score=round(min(1.0, food_area_score), 3),
        green_share=round(min(1.0, green_share), 3),
        red_share=round(min(1.0, red_share), 3),
        yellow_share=round(min(1.0, yellow_share), 3),
        brown_share=round(min(1.0, brown_share), 3),
        white_share=round(min(1.0, white_share), 3),
        orange_share=round(min(1.0, orange_share), 3),
    )


def _mask_share(
    hsv_image: np.ndarray,
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> float:
    import cv2
    import numpy as np

    (h_low, h_high), (s_low, s_high), (v_low, v_high) = bounds
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower, upper)
    return float((mask > 0).mean())
