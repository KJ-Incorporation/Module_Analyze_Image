"""Response schemas for food analysis."""

from pydantic import BaseModel, ConfigDict, Field


class FoodDetectedItemResponse(BaseModel):
    """Detected food component with a coarse confidence score."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class FoodAnalysisNotesResponse(BaseModel):
    """Frontend-friendly notes for food analysis."""

    attention: str
    parfait: str
    progression: str


class FoodMealFeedbackResponse(BaseModel):
    """Frontend-friendly product feedback about the meal itself."""

    attention: str
    parfait: str
    progression: str


class FoodImageAnalysisResponse(BaseModel):
    """Per-image food analysis payload."""

    filename: str
    content_type: str | None = None
    processing_status: str
    food_detected: bool
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    image_width: int | None = Field(default=None, ge=0)
    image_height: int | None = Field(default=None, ge=0)
    meal_label: str | None = None
    estimated_health_profile: str | None = None
    detected_items: list[FoodDetectedItemResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnalyzeFoodResponse(BaseModel):
    """Top-level JSON response returned by the food analysis endpoint."""

    user_id: str | None = None
    meal_context: str | None = None
    locale: str | None = None
    images_processed: int = Field(ge=0)
    food_detected: bool
    meal_label: str | None = None
    detected_items: list[FoodDetectedItemResponse] = Field(default_factory=list)
    estimated_portion_label: str | None = None
    estimated_portion_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    estimated_calories_kcal: int | None = Field(default=None, ge=0)
    estimated_protein_g: int | None = Field(default=None, ge=0)
    estimated_carbs_g: int | None = Field(default=None, ge=0)
    estimated_fat_g: int | None = Field(default=None, ge=0)
    estimated_health_profile: str | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str
    analysis_notes: FoodAnalysisNotesResponse
    meal_feedback: FoodMealFeedbackResponse
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    results: list[FoodImageAnalysisResponse] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "meal_context": "lunch",
                "locale": "fr-FR",
                "images_processed": 1,
                "food_detected": True,
                "meal_label": "chicken_rice_bowl",
                "detected_items": [
                    {"label": "rice", "confidence": 0.82},
                    {"label": "chicken_or_lean_protein", "confidence": 0.78},
                    {"label": "vegetables", "confidence": 0.7},
                ],
                "estimated_portion_label": "medium",
                "estimated_portion_confidence": 0.61,
                "estimated_calories_kcal": 590,
                "estimated_protein_g": 38,
                "estimated_carbs_g": 58,
                "estimated_fat_g": 18,
                "estimated_health_profile": "protein_forward",
                "confidence_score": 0.72,
                "model_version": "heuristic-food-v0.2.0",
                "analysis_notes": {
                    "attention": "Attention: la portion reste approximative a partir d'une photo seule.",
                    "parfait": "Parfait: les aliments principaux du plat ont ete reconnus de facon exploitable.",
                    "progression": "Progression: une photo prise de dessus, avec un cadrage plus propre, ameliorera l'estimation nutritionnelle.",
                },
                "meal_feedback": {
                    "attention": "Attention: la sauce ou les accompagnements peuvent faire varier les calories reelles.",
                    "parfait": "Parfait: ce repas semble plutot riche en proteines avec une base plus equilibree qu'un fast-food classique.",
                    "progression": "Progression: ajoute davantage de legumes ou garde une portion stable de glucides pour un repas encore plus regulier.",
                },
                "warnings": [
                    "Nutrition values are approximate and should not be treated as exact."
                ],
                "recommendations": [
                    "Take one clear top-down photo with the whole plate visible."
                ],
                "results": [
                    {
                        "filename": "meal.jpg",
                        "content_type": "image/jpeg",
                        "processing_status": "success",
                        "food_detected": True,
                        "confidence_score": 0.69,
                        "quality_score": 0.85,
                        "image_width": 1080,
                        "image_height": 1440,
                        "meal_label": "chicken_rice_bowl",
                        "estimated_health_profile": "protein_forward",
                        "detected_items": [
                            {"label": "rice", "confidence": 0.82},
                            {"label": "chicken_or_lean_protein", "confidence": 0.78},
                        ],
                        "warnings": [],
                    }
                ],
            }
        }
    )
