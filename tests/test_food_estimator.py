"""Unit tests for heuristic food estimation."""

from app.services.food_estimator import (
    build_food_meal_feedback,
    estimate_food_from_features,
    merge_food_estimates,
)
from app.services.food_feature_engineering import FoodVisualFeatures


def test_estimate_food_from_features_detects_salad_like_profile() -> None:
    estimate = estimate_food_from_features(
        features=FoodVisualFeatures(
            mean_brightness=0.62,
            mean_saturation=0.71,
            edge_density=0.09,
            colorfulness_score=0.68,
            food_area_score=0.84,
            green_share=0.42,
            red_share=0.08,
            yellow_share=0.09,
            brown_share=0.05,
            white_share=0.06,
            orange_share=0.04,
        ),
        quality_score=0.86,
    )

    assert estimate.food_detected is True
    assert estimate.meal_label == "salad_bowl"
    assert estimate.estimated_health_profile == "fiber_forward"
    assert estimate.estimated_calories_kcal is not None
    assert estimate.detected_items[0].label == "lettuce"


def test_estimate_food_from_features_detects_chicken_rice_like_profile() -> None:
    estimate = estimate_food_from_features(
        features=FoodVisualFeatures(
            mean_brightness=0.58,
            mean_saturation=0.54,
            edge_density=0.12,
            colorfulness_score=0.46,
            food_area_score=0.77,
            green_share=0.09,
            red_share=0.04,
            yellow_share=0.10,
            brown_share=0.18,
            white_share=0.22,
            orange_share=0.03,
        ),
        quality_score=0.82,
    )

    assert estimate.food_detected is True
    assert estimate.meal_label == "chicken_rice_bowl"
    assert estimate.estimated_health_profile == "protein_forward"
    assert estimate.estimated_protein_g is not None
    assert estimate.estimated_protein_g >= 30


def test_estimate_food_from_features_returns_no_food_when_scene_is_too_weak() -> None:
    estimate = estimate_food_from_features(
        features=FoodVisualFeatures(
            mean_brightness=0.94,
            mean_saturation=0.08,
            edge_density=0.01,
            colorfulness_score=0.05,
            food_area_score=0.16,
            green_share=0.01,
            red_share=0.01,
            yellow_share=0.01,
            brown_share=0.01,
            white_share=0.88,
            orange_share=0.01,
        ),
        quality_score=0.72,
    )

    assert estimate.food_detected is False
    assert estimate.meal_label is None
    assert estimate.estimated_calories_kcal is None
    assert estimate.estimated_health_profile is None


def test_estimate_food_from_features_detects_pizza_like_profile() -> None:
    estimate = estimate_food_from_features(
        features=FoodVisualFeatures(
            mean_brightness=0.58,
            mean_saturation=0.69,
            edge_density=0.11,
            colorfulness_score=0.63,
            food_area_score=0.82,
            green_share=0.03,
            red_share=0.16,
            yellow_share=0.19,
            brown_share=0.11,
            white_share=0.14,
            orange_share=0.10,
        ),
        quality_score=0.84,
    )

    assert estimate.food_detected is True
    assert estimate.meal_label == "pizza_plate"
    assert estimate.estimated_health_profile == "indulgent"
    assert estimate.confidence_score is not None
    assert estimate.confidence_score >= 0.5


def test_estimate_food_from_features_detects_burger_like_profile() -> None:
    estimate = estimate_food_from_features(
        features=FoodVisualFeatures(
            mean_brightness=0.55,
            mean_saturation=0.58,
            edge_density=0.12,
            colorfulness_score=0.52,
            food_area_score=0.73,
            green_share=0.04,
            red_share=0.09,
            yellow_share=0.12,
            brown_share=0.22,
            white_share=0.11,
            orange_share=0.05,
        ),
        quality_score=0.81,
    )

    assert estimate.food_detected is True
    assert estimate.meal_label == "burger_meal"
    assert estimate.estimated_health_profile == "indulgent"
    assert estimate.confidence_score is not None
    assert estimate.confidence_score >= 0.5


def test_merge_food_estimates_keeps_best_supported_meal_label() -> None:
    first = estimate_food_from_features(
        features=FoodVisualFeatures(
            mean_brightness=0.57,
            mean_saturation=0.53,
            edge_density=0.11,
            colorfulness_score=0.49,
            food_area_score=0.72,
            green_share=0.08,
            red_share=0.07,
            yellow_share=0.12,
            brown_share=0.16,
            white_share=0.28,
            orange_share=0.05,
        ),
        quality_score=0.81,
    )
    second = estimate_food_from_features(
        features=FoodVisualFeatures(
            mean_brightness=0.59,
            mean_saturation=0.56,
            edge_density=0.12,
            colorfulness_score=0.52,
            food_area_score=0.74,
            green_share=0.1,
            red_share=0.06,
            yellow_share=0.15,
            brown_share=0.19,
            white_share=0.24,
            orange_share=0.04,
        ),
        quality_score=0.84,
    )

    merged = merge_food_estimates([first, second])

    assert merged.food_detected is True
    assert merged.meal_label is not None
    assert merged.estimated_health_profile is not None
    assert merged.confidence_score is not None


def test_build_food_meal_feedback_returns_product_notes() -> None:
    estimate = estimate_food_from_features(
        features=FoodVisualFeatures(
            mean_brightness=0.58,
            mean_saturation=0.54,
            edge_density=0.12,
            colorfulness_score=0.46,
            food_area_score=0.77,
            green_share=0.09,
            red_share=0.04,
            yellow_share=0.10,
            brown_share=0.18,
            white_share=0.22,
            orange_share=0.03,
        ),
        quality_score=0.82,
    )

    attention, parfait, progression = build_food_meal_feedback(estimate)

    assert attention.startswith("Attention:")
    assert parfait.startswith("Parfait:")
    assert progression.startswith("Progression:")
