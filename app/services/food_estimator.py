"""Heuristic food classification and nutrition estimation."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.food_feature_engineering import FoodVisualFeatures


@dataclass(frozen=True, slots=True)
class FoodItemEstimate:
    """Single food item estimate."""

    label: str
    confidence: float


@dataclass(frozen=True, slots=True)
class FoodEstimate:
    """Top-level food estimate for one or more images."""

    food_detected: bool
    meal_label: str | None
    detected_items: list[FoodItemEstimate]
    estimated_portion_label: str | None
    estimated_portion_confidence: float | None
    estimated_calories_kcal: int | None
    estimated_protein_g: int | None
    estimated_carbs_g: int | None
    estimated_fat_g: int | None
    estimated_health_profile: str | None
    confidence_score: float | None
    warnings: list[str]


MEAL_ITEM_MAP = {
    "salad_bowl": ["lettuce", "vegetables", "dressing"],
    "chicken_rice_bowl": ["rice", "chicken_or_lean_protein", "vegetables"],
    "oatmeal_bowl": ["oats", "toppings", "milk_or_yogurt"],
    "greek_yogurt_bowl": ["greek_yogurt", "fruit", "toppings"],
    "fruit_bowl": ["fruit", "mixed_fruit", "fresh_toppings"],
    "egg_meal": ["eggs", "side", "toast_or_garnish"],
    "sushi_plate": ["rice", "fish_or_filling", "seaweed_or_side"],
    "pasta_plate": ["pasta", "sauce", "garnish"],
    "pizza_plate": ["pizza", "cheese", "tomato_sauce"],
    "burger_meal": ["burger_bun", "patty_or_filling", "sauce_or_cheese"],
    "sandwich_wrap": ["bread_or_wrap", "filling", "sauce"],
    "soup_or_curry": ["broth_or_sauce", "protein_or_legumes", "vegetables"],
    "dessert_plate": ["dessert", "topping", "sweet_sauce"],
    "mixed_healthy_plate": ["lean_protein", "starch", "vegetables"],
    "mixed_plate": ["protein", "starch", "vegetables"],
}

MEAL_MACROS = {
    "salad_bowl": {"calories": 420, "protein": 24, "carbs": 26, "fat": 22},
    "chicken_rice_bowl": {"calories": 590, "protein": 38, "carbs": 58, "fat": 18},
    "oatmeal_bowl": {"calories": 410, "protein": 15, "carbs": 58, "fat": 12},
    "greek_yogurt_bowl": {"calories": 360, "protein": 24, "carbs": 34, "fat": 10},
    "fruit_bowl": {"calories": 240, "protein": 4, "carbs": 52, "fat": 3},
    "egg_meal": {"calories": 430, "protein": 28, "carbs": 24, "fat": 22},
    "sushi_plate": {"calories": 520, "protein": 28, "carbs": 62, "fat": 16},
    "pasta_plate": {"calories": 690, "protein": 24, "carbs": 84, "fat": 24},
    "pizza_plate": {"calories": 760, "protein": 30, "carbs": 86, "fat": 31},
    "burger_meal": {"calories": 720, "protein": 34, "carbs": 52, "fat": 39},
    "sandwich_wrap": {"calories": 560, "protein": 28, "carbs": 48, "fat": 24},
    "soup_or_curry": {"calories": 480, "protein": 22, "carbs": 42, "fat": 22},
    "dessert_plate": {"calories": 520, "protein": 7, "carbs": 62, "fat": 24},
    "mixed_healthy_plate": {"calories": 540, "protein": 34, "carbs": 42, "fat": 20},
    "mixed_plate": {"calories": 640, "protein": 34, "carbs": 58, "fat": 24},
}

MEAL_HEALTH_PROFILE_MAP = {
    "salad_bowl": "fiber_forward",
    "chicken_rice_bowl": "protein_forward",
    "oatmeal_bowl": "lean_balanced",
    "greek_yogurt_bowl": "protein_forward",
    "fruit_bowl": "fiber_forward",
    "egg_meal": "protein_forward",
    "sushi_plate": "lean_balanced",
    "pasta_plate": "carb_dense",
    "pizza_plate": "indulgent",
    "burger_meal": "indulgent",
    "sandwich_wrap": "mixed",
    "soup_or_curry": "mixed",
    "dessert_plate": "indulgent",
    "mixed_healthy_plate": "lean_balanced",
    "mixed_plate": "mixed",
}

PORTION_MULTIPLIERS = {
    "small": 0.82,
    "medium": 1.0,
    "large": 1.18,
}


def estimate_food_from_features(
    *,
    features: FoodVisualFeatures,
    quality_score: float | None,
) -> FoodEstimate:
    """Estimate a food category and approximate nutrition from visual features."""

    warnings: list[str] = [
        "Nutrition values are approximate and should not be treated as exact."
    ]
    food_presence_score = _compute_food_presence_score(features, quality_score)
    if food_presence_score < 0.38:
        warnings.append("No plate or food-like scene was detected with enough confidence.")
        return FoodEstimate(
            food_detected=False,
            meal_label=None,
            detected_items=[],
            estimated_portion_label=None,
            estimated_portion_confidence=None,
            estimated_calories_kcal=None,
            estimated_protein_g=None,
            estimated_carbs_g=None,
            estimated_fat_g=None,
            estimated_health_profile=None,
            confidence_score=round(food_presence_score, 3),
            warnings=warnings,
        )

    scores = {
        "salad_bowl": _score_salad(features),
        "chicken_rice_bowl": _score_chicken_rice(features),
        "oatmeal_bowl": _score_oatmeal(features),
        "greek_yogurt_bowl": _score_greek_yogurt(features),
        "fruit_bowl": _score_fruit_bowl(features),
        "egg_meal": _score_egg_meal(features),
        "sushi_plate": _score_sushi(features),
        "pasta_plate": _score_pasta(features),
        "pizza_plate": _score_pizza(features),
        "burger_meal": _score_burger(features),
        "sandwich_wrap": _score_sandwich(features),
        "soup_or_curry": _score_soup_or_curry(features),
        "dessert_plate": _score_dessert(features),
        "mixed_healthy_plate": _score_mixed_healthy_plate(features),
        "mixed_plate": _score_mixed_plate(features),
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    meal_label, top_score = ordered[0]
    if top_score < 0.44:
        meal_label = "mixed_plate"
        top_score = max(top_score, 0.44)
        warnings.append("The meal looked mixed or visually ambiguous, so food classification is broad.")

    portion_label, portion_confidence = _estimate_portion(features)
    detected_items = [
        FoodItemEstimate(
            label=item_label,
            confidence=round(max(0.35, min(0.95, top_score - (index * 0.07))), 3),
        )
        for index, item_label in enumerate(MEAL_ITEM_MAP[meal_label])
    ]
    macros = _estimate_macros(meal_label, portion_label)
    health_profile = _estimate_health_profile(meal_label, macros)

    return FoodEstimate(
        food_detected=True,
        meal_label=meal_label,
        detected_items=detected_items,
        estimated_portion_label=portion_label,
        estimated_portion_confidence=portion_confidence,
        estimated_calories_kcal=macros["calories"],
        estimated_protein_g=macros["protein"],
        estimated_carbs_g=macros["carbs"],
        estimated_fat_g=macros["fat"],
        estimated_health_profile=health_profile,
        confidence_score=round(min(1.0, ((food_presence_score * 0.45) + (top_score * 0.55))), 3),
        warnings=warnings,
    )


def merge_food_estimates(estimates: list[FoodEstimate]) -> FoodEstimate:
    """Merge per-image food estimates into one top-level result."""

    valid_estimates = [estimate for estimate in estimates if estimate.food_detected]
    warnings = _deduplicate_strings(warning for estimate in estimates for warning in estimate.warnings)
    if not valid_estimates:
        return FoodEstimate(
            food_detected=False,
            meal_label=None,
            detected_items=[],
            estimated_portion_label=None,
            estimated_portion_confidence=None,
            estimated_calories_kcal=None,
            estimated_protein_g=None,
            estimated_carbs_g=None,
            estimated_fat_g=None,
            estimated_health_profile=None,
            confidence_score=0.0,
            warnings=warnings,
        )

    by_label: dict[str, list[FoodEstimate]] = {}
    for estimate in valid_estimates:
        by_label.setdefault(estimate.meal_label or "mixed_plate", []).append(estimate)

    best_label, grouped = max(
        by_label.items(),
        key=lambda item: sum(estimate.confidence_score or 0.0 for estimate in item[1]) / len(item[1]),
    )
    representative = max(grouped, key=lambda estimate: estimate.confidence_score or 0.0)
    confidence = round(
        sum((estimate.confidence_score or 0.0) for estimate in grouped) / len(grouped),
        3,
    )

    return FoodEstimate(
        food_detected=True,
        meal_label=best_label,
        detected_items=representative.detected_items,
        estimated_portion_label=representative.estimated_portion_label,
        estimated_portion_confidence=representative.estimated_portion_confidence,
        estimated_calories_kcal=representative.estimated_calories_kcal,
        estimated_protein_g=representative.estimated_protein_g,
        estimated_carbs_g=representative.estimated_carbs_g,
        estimated_fat_g=representative.estimated_fat_g,
        estimated_health_profile=representative.estimated_health_profile,
        confidence_score=confidence,
        warnings=warnings,
    )


def build_food_analysis_notes(estimate: FoodEstimate) -> tuple[str, str, str]:
    """Build attention / parfait / progression notes for scan quality and reliability."""

    if not estimate.food_detected:
        return (
            "Attention: aucun plat n'a ete detecte de facon assez fiable sur les images fournies.",
            "Parfait: la validation des fichiers fonctionne, donc le scan food peut etre relance proprement.",
            "Progression: prends une photo nette, bien eclairee et avec le plat entier dans le cadre pour obtenir une analyse nutritionnelle.",
        )

    attention = "Attention: la portion reste approximative a partir d'une photo seule."
    if estimate.confidence_score is not None and estimate.confidence_score < 0.55:
        attention = "Attention: le plat a ete reconnu, mais la confiance reste moyenne sur la composition exacte."

    parfait = "Parfait: les grandes composantes du repas ont ete reconnues de facon exploitable."
    progression = (
        "Progression: une photo prise de dessus, avec le plat entier bien visible et peu d'arriere-plan, ameliorera l'estimation nutritionnelle."
    )
    return attention, parfait, progression


def build_food_meal_feedback(estimate: FoodEstimate) -> tuple[str, str, str]:
    """Build product-friendly meal feedback oriented to healthy everyday meals."""

    if not estimate.food_detected or not estimate.meal_label:
        return (
            "Attention: l'image ne permet pas encore de tirer une conclusion fiable sur le repas.",
            "Parfait: le flow est pret pour un nouveau scan des que le plat est mieux cadre.",
            "Progression: centre le plat, evite les ombres fortes et garde le repas entier dans le cadre.",
        )

    profile = estimate.estimated_health_profile or "mixed"
    attention_map = {
        "protein_forward": "Attention: la sauce, l'huile ou certains accompagnements peuvent encore faire varier les calories reelles.",
        "lean_balanced": "Attention: la portion reelle reste a surveiller meme si le repas semble globalement equilibre.",
        "fiber_forward": "Attention: les toppings ou sauces sucrees peuvent changer vite le profil nutritionnel reel.",
        "carb_dense": "Attention: ce repas semble plutot riche en glucides, donc la portion peut vite peser dans le total calorique.",
        "fat_dense": "Attention: ce repas semble assez dense en lipides, surtout si la cuisson ou la sauce sont genereuses.",
        "indulgent": "Attention: ce repas parait plus dense en calories et en lipides qu'un repas du quotidien plus simple.",
        "mixed": "Attention: le repas semble melange, donc une partie des calories peut varier selon la portion et les accompagnements.",
    }
    parfait_map = {
        "protein_forward": "Parfait: ce repas parait riche en proteines et plutot coherent pour soutenir la satiete.",
        "lean_balanced": "Parfait: l'ensemble parait assez equilibre entre energie, proteines et densite calorique.",
        "fiber_forward": "Parfait: la base du repas semble plutot fraiche et interessante pour ajouter du volume sans trop charger.",
        "carb_dense": "Parfait: la source principale d'energie est bien identifiee, ce qui rend le repas lisible pour le suivi.",
        "fat_dense": "Parfait: la structure du plat est assez claire, donc le backend renvoie tout de meme une estimation exploitable.",
        "indulgent": "Parfait: le type de repas est bien reconnu, ce qui permet au suivi de rester simple meme sur un repas plaisir.",
        "mixed": "Parfait: les composantes majeures du repas ont ete reconnues de facon suffisamment stable pour un retour produit.",
    }
    progression_map = {
        "protein_forward": "Progression: ajoute encore plus de legumes ou garde une portion stable de glucides pour un repas encore plus regulier.",
        "lean_balanced": "Progression: garde ce type de structure simple et stable pour faciliter le suivi dans l'app.",
        "fiber_forward": "Progression: ajoute une source de proteines plus marquee si tu veux un repas plus complet et plus rassasiant.",
        "carb_dense": "Progression: ajoute une source de proteines et davantage de legumes pour mieux equilibrer l'assiette.",
        "fat_dense": "Progression: une cuisson plus simple ou une sauce plus legere rendrait le repas plus facile a integrer sur la journee.",
        "indulgent": "Progression: garde ce type de repas comme plaisir ponctuel ou compense avec une portion plus raisonnable et un accompagnement plus simple.",
        "mixed": "Progression: une photo plus propre ou une assiette moins chargee rendra l'analyse plus precise et le feedback plus utile.",
    }
    return attention_map[profile], parfait_map[profile], progression_map[profile]


def build_food_recommendations(estimate: FoodEstimate) -> list[str]:
    """Generate pragmatic frontend recommendations for food analysis."""

    recommendations: list[str] = []
    if not estimate.food_detected:
        recommendations.append("Take one clear top-down photo with the whole plate visible.")
        recommendations.append("Avoid strong shadows and keep the meal centered in the frame.")
        return recommendations

    if estimate.estimated_portion_confidence is not None and estimate.estimated_portion_confidence < 0.6:
        recommendations.append("Include the whole plate in frame to improve portion estimation.")
    if estimate.confidence_score is not None and estimate.confidence_score < 0.6:
        recommendations.append("Use brighter lighting and avoid busy backgrounds around the meal.")
    if estimate.estimated_health_profile in {"indulgent", "carb_dense"}:
        recommendations.append("A side of vegetables or a smaller sauce portion would make the meal easier to track.")
    elif estimate.estimated_health_profile in {"protein_forward", "lean_balanced"}:
        recommendations.append("This meal pattern looks easier to keep consistent in a body-composition routine.")
    return recommendations


def _compute_food_presence_score(features: FoodVisualFeatures, quality_score: float | None) -> float:
    quality = quality_score or 0.0
    return min(
        1.0,
        (features.food_area_score * 0.35)
        + (features.mean_saturation * 0.20)
        + (features.colorfulness_score * 0.20)
        + (min(1.0, features.edge_density / 0.18) * 0.15)
        + (quality * 0.10),
    )


def _score_salad(features: FoodVisualFeatures) -> float:
    score = (
        (features.green_share * 0.50)
        + (features.mean_saturation * 0.20)
        + (features.colorfulness_score * 0.20)
        + (features.food_area_score * 0.10)
    )
    if features.green_share >= 0.30 and features.mean_saturation >= 0.55:
        score += 0.08
    return min(1.0, score)


def _score_chicken_rice(features: FoodVisualFeatures) -> float:
    score = (
        (features.white_share * 0.34)
        + (features.brown_share * 0.26)
        + (features.yellow_share * 0.14)
        + (features.food_area_score * 0.22)
        + (features.edge_density * 0.10)
        + (features.green_share * 0.10)
    )
    if features.white_share >= 0.18 and features.brown_share >= 0.12:
        score += 0.12
    if features.red_share <= 0.06 and features.white_share >= 0.18:
        score += 0.06
    return min(1.0, score)


def _score_oatmeal(features: FoodVisualFeatures) -> float:
    low_edge_bonus = max(0.0, 1.0 - (features.edge_density / 0.20))
    score = (
        (features.white_share * 0.20)
        + (features.brown_share * 0.26)
        + (features.mean_brightness * 0.10)
        + (low_edge_bonus * 0.20)
        + (features.food_area_score * 0.14)
        + (features.mean_saturation * 0.10)
    )
    if features.brown_share >= 0.12 and features.white_share >= 0.10:
        score += 0.06
    return min(1.0, score)


def _score_greek_yogurt(features: FoodVisualFeatures) -> float:
    score = (
        (features.white_share * 0.32)
        + (features.red_share * 0.12)
        + (features.yellow_share * 0.12)
        + (features.mean_brightness * 0.14)
        + (features.food_area_score * 0.14)
        + (features.colorfulness_score * 0.10)
    )
    if features.white_share >= 0.18 and (features.red_share >= 0.06 or features.yellow_share >= 0.08):
        score += 0.08
    return min(1.0, score)


def _score_fruit_bowl(features: FoodVisualFeatures) -> float:
    score = (
        (features.red_share * 0.20)
        + (features.yellow_share * 0.22)
        + (features.green_share * 0.18)
        + (features.colorfulness_score * 0.20)
        + (features.mean_saturation * 0.12)
        + (features.food_area_score * 0.08)
    )
    if features.colorfulness_score >= 0.55 and features.mean_saturation >= 0.55:
        score += 0.08
    return min(1.0, score)


def _score_egg_meal(features: FoodVisualFeatures) -> float:
    score = (
        (features.yellow_share * 0.28)
        + (features.white_share * 0.24)
        + (features.food_area_score * 0.18)
        + (features.edge_density * 0.12)
        + (features.mean_brightness * 0.10)
        + (features.brown_share * 0.08)
    )
    if features.yellow_share >= 0.12 and features.white_share >= 0.10:
        score += 0.08
    return min(1.0, score)


def _score_sushi(features: FoodVisualFeatures) -> float:
    score = (
        (features.white_share * 0.24)
        + (features.red_share * 0.14)
        + (features.orange_share * 0.12)
        + (features.edge_density * 0.20)
        + (features.food_area_score * 0.16)
        + (features.mean_saturation * 0.10)
    )
    if features.white_share >= 0.18 and features.edge_density >= 0.09:
        score += 0.07
    return min(1.0, score)


def _score_pasta(features: FoodVisualFeatures) -> float:
    return min(
        1.0,
        (features.yellow_share * 0.32)
        + (features.red_share * 0.18)
        + (features.orange_share * 0.18)
        + (features.food_area_score * 0.17)
        + (features.mean_saturation * 0.15),
    )


def _score_pizza(features: FoodVisualFeatures) -> float:
    score = (
        (features.red_share * 0.40)
        + (features.yellow_share * 0.42)
        + (features.white_share * 0.20)
        + (features.orange_share * 0.16)
        + (features.edge_density * 0.24)
        + (features.food_area_score * 0.18)
    )
    if features.red_share >= 0.10 and features.yellow_share >= 0.12:
        score += 0.12
    if features.white_share >= 0.10 and features.edge_density >= 0.08:
        score += 0.08
    return min(1.0, score)


def _score_burger(features: FoodVisualFeatures) -> float:
    score = (
        (features.brown_share * 0.42)
        + (features.yellow_share * 0.24)
        + (features.red_share * 0.16)
        + (features.white_share * 0.14)
        + (features.edge_density * 0.22)
        + (features.food_area_score * 0.18)
    )
    if features.brown_share >= 0.18 and features.yellow_share >= 0.08:
        score += 0.12
    if features.edge_density >= 0.10 and features.food_area_score >= 0.60:
        score += 0.08
    return min(1.0, score)


def _score_sandwich(features: FoodVisualFeatures) -> float:
    return min(
        1.0,
        (features.brown_share * 0.30)
        + (features.white_share * 0.18)
        + (features.edge_density * 0.22)
        + (features.food_area_score * 0.18)
        + (features.yellow_share * 0.12),
    )


def _score_soup_or_curry(features: FoodVisualFeatures) -> float:
    low_edge_bonus = max(0.0, 1.0 - (features.edge_density / 0.22))
    return min(
        1.0,
        (features.orange_share * 0.25)
        + (features.red_share * 0.20)
        + (features.yellow_share * 0.18)
        + (low_edge_bonus * 0.22)
        + (features.food_area_score * 0.15),
    )


def _score_dessert(features: FoodVisualFeatures) -> float:
    return min(
        1.0,
        (features.brown_share * 0.24)
        + (features.white_share * 0.18)
        + (features.red_share * 0.14)
        + (features.mean_brightness * 0.14)
        + (features.food_area_score * 0.15)
        + (features.colorfulness_score * 0.15),
    )


def _score_mixed_healthy_plate(features: FoodVisualFeatures) -> float:
    score = (
        (features.green_share * 0.24)
        + (features.brown_share * 0.14)
        + (features.white_share * 0.12)
        + (features.colorfulness_score * 0.18)
        + (features.food_area_score * 0.18)
        + (features.edge_density * 0.08)
        + (features.mean_saturation * 0.06)
    )
    if features.green_share >= 0.12 and (features.brown_share >= 0.10 or features.white_share >= 0.14):
        score += 0.06
    return min(1.0, score)


def _score_mixed_plate(features: FoodVisualFeatures) -> float:
    return min(
        1.0,
        (features.food_area_score * 0.22)
        + (features.colorfulness_score * 0.18)
        + (features.edge_density * 0.14)
        + (features.mean_saturation * 0.12)
        + ((features.green_share + features.brown_share + features.white_share) * 0.06),
    )


def _estimate_portion(features: FoodVisualFeatures) -> tuple[str, float]:
    score = min(
        1.0,
        (features.food_area_score * 0.65)
        + (min(1.0, features.edge_density / 0.16) * 0.15)
        + (features.colorfulness_score * 0.20),
    )
    if score < 0.42:
        return "small", round(max(0.35, score), 3)
    if score < 0.7:
        return "medium", round(score, 3)
    return "large", round(score, 3)


def _estimate_macros(meal_label: str, portion_label: str | None) -> dict[str, int]:
    base = MEAL_MACROS[meal_label]
    multiplier = PORTION_MULTIPLIERS.get(portion_label or "medium", 1.0)
    return {key: int(round(value * multiplier)) for key, value in base.items()}


def _estimate_health_profile(meal_label: str, macros: dict[str, int]) -> str:
    profile = MEAL_HEALTH_PROFILE_MAP.get(meal_label, "mixed")
    if profile != "mixed":
        return profile

    protein = macros["protein"]
    carbs = macros["carbs"]
    fat = macros["fat"]
    calories = macros["calories"]
    if protein >= 30 and calories <= 620:
        return "protein_forward"
    if fat >= 30:
        return "fat_dense"
    if carbs >= 65:
        return "carb_dense"
    if calories <= 520 and fat <= 18:
        return "lean_balanced"
    return "mixed"


def _deduplicate_strings(values) -> list[str]:
    return list(dict.fromkeys(values))
