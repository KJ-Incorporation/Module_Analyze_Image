"""Response schemas for the analysis endpoint."""

from pydantic import BaseModel, ConfigDict, Field


class BoundingBoxResponse(BaseModel):
    """Pixel bounding box around the detected body."""

    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=0)
    y_max: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)


class LandmarkResponse(BaseModel):
    """Body landmark represented in pixel coordinates."""

    index: int = Field(ge=0)
    name: str
    x_px: float | None = None
    y_px: float | None = None
    visibility: float | None = Field(default=None, ge=0.0, le=1.0)
    presence: float | None = Field(default=None, ge=0.0, le=1.0)


class BodyRegionStatusResponse(BaseModel):
    """Checklist item describing whether a body region looks visible."""

    key: str
    label: str
    visible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    taken_into_account: bool = False


class AnalysisNotesResponse(BaseModel):
    """One note block with a title and a message."""

    title: str
    message: str


class AnalysisNotesBlockResponse(BaseModel):
    """Frontend-friendly notes split into three product sections."""

    attention: AnalysisNotesResponse
    parfait: AnalysisNotesResponse
    progression: AnalysisNotesResponse


class AnalysisBlocksResponse(BaseModel):
    """Expanded premium blocks for richer body-scan product screens."""

    overview: AnalysisNotesResponse
    truth: AnalysisNotesResponse
    strength: AnalysisNotesResponse
    limitation: AnalysisNotesResponse
    next_focus: AnalysisNotesResponse
    scan_quality: AnalysisNotesResponse


class SomatotypeScoresResponse(BaseModel):
    """Heuristic scores for the three classic somatotype buckets."""

    ectomorph: float = Field(ge=0.0, le=1.0)
    mesomorph: float = Field(ge=0.0, le=1.0)
    endomorph: float = Field(ge=0.0, le=1.0)


class SomatotypeResponse(BaseModel):
    """Estimated somatotype summary returned to the frontend."""

    primary: str | None = None
    secondary: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    scores: SomatotypeScoresResponse | None = None
    notes: str


class ScanProfileResponse(BaseModel):
    """Premium scan profile derived from the body scan heuristics."""

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


class ImageAnalysisResponse(BaseModel):
    """Per-image analysis payload."""

    filename: str
    content_type: str | None = None
    processing_status: str
    body_detected: bool
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    usable_for_body_fat_estimation: bool = False
    image_width: int | None = Field(default=None, ge=0)
    image_height: int | None = Field(default=None, ge=0)
    bbox: BoundingBoxResponse | None = None
    landmarks: list[LandmarkResponse] = Field(default_factory=list)
    analyzed_regions: list[BodyRegionStatusResponse] = Field(default_factory=list)
    estimated_shoulder_width_px: float | None = Field(default=None, ge=0.0)
    estimated_hip_width_px: float | None = Field(default=None, ge=0.0)
    estimated_waist_width_px: float | None = Field(default=None, ge=0.0)
    estimated_waist_to_hip_ratio: float | None = Field(default=None, ge=0.0)
    posture_summary: str | None = None
    warnings: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    """Top-level JSON response returned by the API."""

    user_id: str | None = None
    sex: str | None = None
    age: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    images_processed: int = Field(ge=0)
    bmi: float | None = Field(default=None, ge=0.0)
    estimated_body_fat_percent: float | None = None
    estimated_fat_mass_kg: float | None = None
    estimated_lean_mass_kg: float | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str
    estimated_body_fat_note: str
    analysis_feedback: str
    coaching_feedback: str
    analysis_notes: AnalysisNotesBlockResponse
    analysis_blocks: AnalysisBlocksResponse
    scan_profile: ScanProfileResponse
    somatotype: SomatotypeResponse
    overall_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    analyzed_regions_summary: list[BodyRegionStatusResponse] = Field(default_factory=list)
    results: list[ImageAnalysisResponse] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "sex": "female",
                "age": 31,
                "height_cm": 168.0,
                "weight_kg": 63.5,
                "images_processed": 2,
                "bmi": 22.5,
                "estimated_body_fat_percent": 27.4,
                "estimated_fat_mass_kg": 17.4,
                "estimated_lean_mass_kg": 46.1,
                "confidence_score": 0.71,
                "model_version": "heuristic-bodyfat-v0.3.0",
                "estimated_body_fat_note": (
                    "Image-based body fat is an estimate only and must not be treated as a medical measurement."
                ),
                "analysis_feedback": (
                    "Good analysis base. Image quality looks solid overall. Reliable zones: Tete, Torse, Taille, "
                    "and Hanches. Weak or missing zones: Mollet / jambe gauche. "
                    "The body fat estimate should still be treated as directional only."
                ),
                "coaching_feedback": (
                    "Good overall composition. Mild abdominal fat retention is still likely. "
                    "At an illustrative pace of about 0.5 kg per week, you could approach 24% estimated body fat "
                    "around May 2026."
                ),
                "analysis_notes": {
                    "attention": {
                        "title": "Attention",
                        "message": (
                            "Le point a surveiller reste surtout la taille, qui parait encore un peu plus chargee "
                            "que le reste de la silhouette."
                        ),
                    },
                    "parfait": {
                        "title": "Parfait",
                        "message": (
                            "Le scan est propre et la lecture du physique est deja assez coherente."
                        ),
                    },
                    "progression": {
                        "title": "Progression",
                        "message": (
                            "Si le rythme reste regulier autour de 0.5 kg par semaine, tu pourrais te rapprocher "
                            "de 24% de body fat estime autour de mai 2026."
                        ),
                    },
                },
                "analysis_blocks": {
                    "overview": {
                        "title": "Lecture globale",
                        "message": "Le scan est suffisamment propre pour une lecture exploitable. La structure generale parait equilibree, avec une marge qui reste surtout visible autour de la taille.",
                    },
                    "truth": {
                        "title": "Le constat principal",
                        "message": "Le scan indique surtout que la zone taille reste encore celle qui freine le plus un rendu plus sec.",
                    },
                    "strength": {
                        "title": "Point fort du scan",
                        "message": "Le scan est suffisamment clair pour soutenir un retour plus dur et plus utile.",
                    },
                    "limitation": {
                        "title": "Frein principal",
                        "message": "La retention autour de la taille reste ce qui freine le plus le rendu global.",
                    },
                    "next_focus": {
                        "title": "Priorite physique",
                        "message": "Le levier le plus rentable maintenant reste de continuer a faire descendre la zone qui garde encore le plus de gras residuel.",
                    },
                    "scan_quality": {
                        "title": "Niveau de fiabilite",
                        "message": "Le scan est classe high avec une couverture front_and_side_available et une posture neutral.",
                    },
                },
                "scan_profile": {
                    "reliability_level": "high",
                    "confidence_label": "Tres fiable",
                    "confidence_message": "Le scan est tres fiable: les angles sont bons, la posture est stable et la lecture globale tient bien.",
                    "definition_level": "moderate_to_good",
                    "fat_distribution": "slightly_central",
                    "frame_assessment": "balanced_frame",
                    "view_coverage": "front_and_side_available",
                    "pose_quality": "neutral",
                    "scan_readiness": "ready_for_actionable_feedback",
                    "best_next_focus": "continue_improving_definition",
                    "dominant_strength": "scan_clarity",
                    "dominant_limitation": "central_fat",
                    "summary": "Le scan est suffisamment propre pour une lecture exploitable. La structure parait deja correcte, avec une marge qui semble surtout se jouer sur la definition autour de la taille."
                },
                "somatotype": {
                    "primary": "mesomorph",
                    "secondary": "endomorph",
                    "confidence": 0.72,
                    "scores": {
                        "ectomorph": 0.31,
                        "mesomorph": 0.78,
                        "endomorph": 0.44
                    },
                    "notes": (
                        "Estime a partir des proportions visibles du corps et d'heuristiques liees au body fat uniquement. "
                        "Il s'agit d'une etiquette approximative et non medicale."
                    )
                },
                "overall_quality_score": 0.79,
                "warnings": [
                    "Estimated body fat is derived from demographics and image-based proxies, not direct measurement."
                ],
                "recommendations": [
                    "Capture at least one sharp image with shoulders and hips fully visible.",
                    "Provide sex, age, height_cm, and weight_kg to improve body fat estimation.",
                ],
                "analyzed_regions_summary": [
                    {
                        "key": "head",
                        "label": "Tete",
                        "visible": True,
                        "confidence": 0.98,
                        "taken_into_account": True,
                    },
                    {
                        "key": "left_lower_leg",
                        "label": "Mollet / jambe gauche",
                        "visible": False,
                        "confidence": 0.18,
                        "taken_into_account": False,
                    },
                ],
                "results": [
                    {
                        "filename": "front.jpg",
                        "content_type": "image/jpeg",
                        "processing_status": "success",
                        "body_detected": True,
                        "confidence_score": 0.87,
                        "quality_score": 0.84,
                        "usable_for_body_fat_estimation": True,
                        "image_width": 1080,
                        "image_height": 1440,
                        "bbox": {
                            "x_min": 300,
                            "y_min": 120,
                            "x_max": 790,
                            "y_max": 1380,
                            "width": 490,
                            "height": 1260,
                        },
                        "landmarks": [
                            {
                                "index": 11,
                                "name": "left_shoulder",
                                "x_px": 402.4,
                                "y_px": 320.8,
                                "visibility": 0.98,
                                "presence": 0.99,
                            }
                        ],
                        "analyzed_regions": [
                            {
                                "key": "head",
                                "label": "Tete",
                                "visible": True,
                                "confidence": 0.97,
                                "taken_into_account": True,
                            },
                            {
                                "key": "left_lower_leg",
                                "label": "Mollet / jambe gauche",
                                "visible": False,
                                "confidence": 0.18,
                                "taken_into_account": False,
                            }
                        ],
                        "estimated_shoulder_width_px": 208.1,
                        "estimated_hip_width_px": 196.4,
                        "estimated_waist_width_px": 181.0,
                        "estimated_waist_to_hip_ratio": 0.92,
                        "posture_summary": "upright / balanced",
                        "warnings": [],
                    }
                ],
            }
        }
    )
