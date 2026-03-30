# Weighty Vision Module

Python FastAPI service for Weighty's computer vision MVP. The service receives one or more photos, validates and decodes them with OpenCV, detects body landmarks with MediaPipe Pose Landmarker, engineers visual proxy features, and returns a JSON response centered on body fat estimation.

## What Changed

The project no longer treats image metrics as the final product. The core flow is now:

1. image validation and quality checks
2. pose landmark extraction
3. visual feature engineering
4. body fat inference from demographics plus visual proxies

This keeps FastAPI, OpenCV, and MediaPipe, but makes body fat estimation the top-level business outcome.

## Current Architecture

```text
app/
  main.py
  api/
    routes.py
  core/
    config.py
  schemas/
    request.py
    response.py
  services/
    body_metrics.py
    bodyfat_estimator.py
    feature_engineering.py
    image_loader.py
    inference_pipeline.py
    pose_estimator.py
    quality_checks.py
tests/
  test_body_metrics.py
  test_bodyfat_estimator.py
  test_feature_engineering.py
requirements.txt
README.md
```

## Core Design Decisions

- `bmi` is computed only from `height_cm` and `weight_kg`.
- `estimated_body_fat_percent` is an estimate only. It must never be presented as a medical truth or clinical measurement.
- No real-world body widths in centimeters are inferred from images. All image-derived dimensions remain proxies in pixels or normalized ratios.
- The current `bodyfat_estimator.py` uses a documented heuristic model versioned as `heuristic-bodyfat-v0.3.0`.
- The heuristic combines `age`, `sex`, `height_cm`, `weight_kg`, and visual proxies from successful images.
- If the inputs are incomplete or visual quality is weak, the API returns `null` estimates or a low `confidence_score`.

## Request Contract

### `POST /analyze`

Multipart fields:

- `images`: one or more JPEG, PNG, or WEBP images
- `sex`: optional, preferred input for body fat estimation
- `gender`: optional legacy alias supported for compatibility
- `age`: optional
- `height_cm`: optional
- `weight_kg`: optional
- `user_id`: optional

### `POST /analyze-food`

Multipart fields:

- `images`: one or more JPEG, PNG, or WEBP meal images
- `user_id`: optional
- `meal_context`: optional, for example `breakfast`, `lunch`, `dinner`, `snack`
- `locale`: optional, defaults to `fr-FR`

## API Analyze Cheat Sheet

Use `multipart/form-data` on `POST /analyze`.

### Props to send

- `images`
- `user_id`
- `age`
- `sex`
- `height_cm`
- `weight_kg`

You can also send `gender` instead of `sex`, but `sex` is the preferred field.

### What is required in practice

- Always required:
  - `images`
- Required if you want `bmi`:
  - `height_cm`
  - `weight_kg`
- Required if you want `estimated_body_fat_percent`:
  - `images`
  - `age`
  - `sex`
  - `height_cm`
  - `weight_kg`

### Important limits

- Max images per request: `5`
- Supported image types:
  - `image/jpeg`
  - `image/png`
  - `image/webp`
- Max file size per image: `10 MB`

### Important behavior

- The API can analyze `1` to `5` images in the same request.
- Empty fields should not be sent.
- `bmi` is computed only from `height_cm` and `weight_kg`.
- `estimated_body_fat_percent` is an estimate only, never a medical truth.
- If images are weak, blurry, cropped, or incomplete, confidence can drop and some estimates can return `null`.
- Images are processed in memory by the backend and are not stored by the module.

### Main response fields to read

- `bmi`
- `estimated_body_fat_percent`
- `estimated_fat_mass_kg`
- `estimated_lean_mass_kg`
- `confidence_score`
- `warnings`
- `model_version`
- `analysis_feedback`
- `coaching_feedback`
- `analysis_notes`
- `analysis_blocks`
- `scan_profile`
- `scan_profile.confidence_label`
- `scan_profile.confidence_message`
- `somatotype`
- `analyzed_regions_summary`

### Front-friendly zone summary

- Read `analyzed_regions_summary` first on the frontend.
- Each item returns:
  - `key`
  - `label`
  - `confidence`
  - `taken_into_account`
- Practical frontend rule:
  - if `confidence < 0.55`, treat the zone as weak and display it in red
  - if `taken_into_account = true`, the zone was strong enough to count

### Per-image fields to read

- `results` is still returned for debug, QA, and image-level troubleshooting.
- In the mobile UI, you usually do not need to render every per-image block.

## Response Contract

Top-level fields returned by the API now include:

- `bmi`
- `estimated_body_fat_percent`
- `estimated_fat_mass_kg`
- `estimated_lean_mass_kg`
- `confidence_score`
- `warnings`
- `model_version`
- `analysis_feedback`
- `coaching_feedback`
- `analysis_notes`
- `analysis_blocks`
- `somatotype`

The response still includes per-image analysis details so the mobile app can inspect partial failures and image quality.
For the main product UI, prefer the top-level `analyzed_regions_summary`.

For the food flow, the main product UI should now prefer:

- `meal_label`
- `estimated_health_profile`
- `analysis_notes`
- `scan_profile`
- `meal_feedback`
- `detected_items`
- `confidence_score`

## Installation

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download a MediaPipe Pose Landmarker model:

```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" `
  -OutFile "models/pose_landmarker_lite.task"
```

If the model lives elsewhere:

```powershell
$env:WEIGHTY_POSE_MODEL_PATH="C:\path\to\pose_landmarker_lite.task"
```

## Run Locally

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Available URLs:

- `http://localhost:8000`
- `http://localhost:8000/docs`
- `http://localhost:8000/test-ui`
- `http://localhost:8000/test-food-ui`

## Deploy on Render

This repo now ships with a Docker-based Render setup so MediaPipe has the native Linux libraries it needs at runtime.

Included files:

- `render.yaml`
- `Dockerfile`
- `.dockerignore`

Render setup summary:

- Push this repo to GitHub
- In Render, create a new `Blueprint` deployment from the repo
- Render will build the image from `Dockerfile`
- The container installs the native libraries required by MediaPipe, including `libgles2`
- The MediaPipe `.task` model is downloaded during the Docker build
- The service starts with `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}`
- Health check path: `/health`

Important:

- This Docker setup is the recommended Render path for this project because the native Python runtime on Render can miss system libraries required by MediaPipe.
- The model file is still not committed because `models/` is ignored.
- `WEIGHTY_POSE_MODEL_PATH` is set automatically inside the container to `/opt/weighty/models/pose_landmarker_lite.task`.
- After deployment, the main URLs are:
  - `/docs`
  - `/test-ui`
  - `/test-food-ui`

## Example curl Request

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "images=@./samples/front.jpg" \
  -F "images=@./samples/side.jpg" \
  -F "sex=female" \
  -F "age=31" \
  -F "height_cm=168" \
  -F "weight_kg=63.5" \
  -F "user_id=user_12345"
```

## Example curl Request for Food

```bash
curl -X POST "http://localhost:8000/analyze-food" \
  -F "images=@./samples/meal.jpg" \
  -F "user_id=user_12345" \
  -F "meal_context=lunch" \
  -F "locale=fr-FR"
```

## Example JSON Response

```json
{
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
  "estimated_body_fat_note": "Image-based body fat is an estimate only and must not be treated as a medical measurement.",
  "analysis_feedback": "La base d'analyse est solide. La qualite des images est globalement bonne. Zones les plus fiables : Tete et Hanches. Zones encore faibles ou partielles : Mollet / jambe gauche. L'estimation du body fat doit rester lue comme une indication, pas comme une mesure medicale.",
  "coaching_feedback": "La composition generale est bonne, avec encore une legere retention probable autour de la taille. A un rythme illustratif d'environ 0.5 kg par semaine, tu pourrais te rapprocher de 24% de body fat estime autour de mai 2026.",
  "analysis_notes": {
    "attention": {
      "title": "Taille a surveiller",
      "message": "Le point a surveiller reste surtout la taille, qui parait encore un peu plus chargee que le reste de la silhouette."
    },
    "parfait": {
      "title": "Scan propre et lisible",
      "message": "Le scan est propre et la lecture du physique est deja assez coherente."
    },
    "progression": {
      "title": "Priorite definition",
      "message": "Si le rythme reste regulier autour de 0.5 kg par semaine, tu pourrais te rapprocher de 24% de body fat estime autour de mai 2026."
    }
  },
  "analysis_blocks": {
    "overview": {
      "title": "Lecture globale",
      "message": "Le scan est suffisamment propre pour une lecture exploitable. La structure generale parait equilibree, avec une marge surtout visible autour de la taille."
    },
    "truth": {
      "title": "Le constat principal",
      "message": "Le scan indique surtout que la zone taille reste encore celle qui freine le plus un rendu plus sec."
    },
    "strength": {
      "title": "Point fort du scan",
      "message": "Le scan est suffisamment clair pour soutenir un retour plus dur et plus utile."
    },
    "limitation": {
      "title": "Frein principal",
      "message": "La retention autour de la taille reste ce qui freine le plus le rendu global."
    },
    "next_focus": {
      "title": "Priorite physique",
      "message": "Le levier le plus rentable maintenant reste de continuer a faire descendre la zone qui garde encore le plus de gras residuel."
    },
    "scan_quality": {
      "title": "Niveau de fiabilite",
      "message": "Le scan est tres fiable: les angles sont bons, la posture est stable et la lecture globale tient bien. La couverture actuelle est front_and_side_available et la posture est evaluee comme neutral."
    }
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
    "summary": "Le scan est suffisamment propre pour une lecture exploitable. Le scan est suffisamment clair pour appuyer un retour plus direct. La structure generale parait equilibree. La definition visuelle parait deja correcte. La marge la plus visible semble legerement centree autour de la taille. La limite principale semble rester une retention plus visible autour de la taille. Le meilleur levier semble etre de continuer a gagner en definition."
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
    "notes": "Estime a partir des proportions visibles du corps et d'heuristiques liees au body fat uniquement. Il s'agit d'une etiquette approximative et non medicale."
  },
  "overall_quality_score": 0.79,
  "warnings": [
    "Estimated body fat is derived from demographics and image-based proxies, not direct measurement."
  ],
  "recommendations": [
    "Use at least one sharp full-body photo to improve body fat estimation confidence."
  ],
  "analyzed_regions_summary": [
    {
      "key": "head",
      "label": "Tete",
      "visible": true,
      "confidence": 0.994,
      "taken_into_account": true
    },
    {
      "key": "left_lower_leg",
      "label": "Mollet / jambe gauche",
      "visible": false,
      "confidence": 0.18,
      "taken_into_account": false
    }
  ],
  "results": [
    {
      "filename": "front.jpg",
      "content_type": "image/jpeg",
      "processing_status": "success",
      "body_detected": true,
      "confidence_score": 0.87,
      "quality_score": 0.84,
      "usable_for_body_fat_estimation": true,
      "image_width": 1080,
      "image_height": 1440,
      "bbox": {
        "x_min": 296,
        "y_min": 116,
        "x_max": 786,
        "y_max": 1384,
        "width": 490,
        "height": 1268
      },
      "landmarks": [
        {
          "index": 11,
          "name": "left_shoulder",
          "x_px": 403.2,
          "y_px": 319.7,
          "visibility": 0.9821,
          "presence": 0.9912
        }
      ],
      "estimated_shoulder_width_px": 208.11,
      "estimated_hip_width_px": 196.44,
      "estimated_waist_width_px": 180.73,
      "estimated_waist_to_hip_ratio": 0.92,
      "posture_summary": "upright / balanced",
      "warnings": []
    }
  ]
}
```

## Example JSON Response for Food

```json
  {
    "user_id": "user_12345",
    "meal_context": "lunch",
    "locale": "fr-FR",
    "images_processed": 1,
    "food_detected": true,
    "meal_label": "chicken_rice_bowl",
    "detected_items": [
      {
        "label": "rice",
        "confidence": 0.82
      },
      {
        "label": "chicken_or_lean_protein",
        "confidence": 0.78
      },
      {
        "label": "vegetables",
        "confidence": 0.71
      }
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
      "attention": "Attention: la portion reste approximate a partir d'une photo seule.",
      "parfait": "Parfait: les grandes composantes du repas ont ete reconnues de facon exploitable.",
      "progression": "Progression: une photo prise de dessus, avec le plat entier bien visible et peu d'arriere-plan, ameliorera l'estimation nutritionnelle."
    },
    "meal_feedback": {
      "attention": "Attention: la sauce, l'huile ou certains accompagnements peuvent encore faire varier les calories reelles.",
      "parfait": "Parfait: ce repas parait riche en proteines et plutot coherent pour soutenir la satiete.",
      "progression": "Progression: ajoute encore plus de legumes ou garde une portion stable de glucides pour un repas encore plus regulier."
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
        "food_detected": true,
        "confidence_score": 0.74,
        "quality_score": 0.85,
        "image_width": 1080,
        "image_height": 1440,
        "meal_label": "chicken_rice_bowl",
        "estimated_health_profile": "protein_forward",
        "detected_items": [
          {
            "label": "rice",
            "confidence": 0.82
          },
          {
            "label": "chicken_or_lean_protein",
            "confidence": 0.78
          }
        ],
        "warnings": []
      }
    ]
  }
```

### Food flow practical fields

For the app, the most useful fields to read first are:

- `meal_label`
- `estimated_health_profile`
- `estimated_calories_kcal`
- `estimated_protein_g`
- `estimated_carbs_g`
- `estimated_fat_g`
- `analysis_notes`
- `meal_feedback`
- `confidence_score`

## Inference Details

`feature_engineering.py` derives normalized visual proxies from each usable image:

- `estimated_waist_to_hip_ratio`
- `estimated_shoulder_to_hip_ratio`
- `estimated_waist_to_bbox_height_ratio`
- `estimated_hip_to_bbox_height_ratio`
- `estimated_shoulder_to_bbox_height_ratio`

`bodyfat_estimator.py` combines those proxies with:

- `age`
- `sex`
- `height_cm`
- `weight_kg`
- BMI derived from height and weight

The current estimator is heuristic. It is designed for MVP behavior, not clinical accuracy.

## Tests

Included unit tests cover:

- BMI calculation
- pose-derived visual metrics
- visual feature engineering
- body fat estimator behavior with complete and incomplete data

Run them with:

```powershell
python -m pytest
```

## Known Limits

- A missing or unsupported `sex`, missing `age`, or missing BMI inputs prevents body fat estimation.
- If no image yields reliable torso features, `estimated_body_fat_percent` returns `null`.
- Full body visibility and image sharpness materially affect the final confidence score.
- The current estimator is a versioned heuristic placeholder until a trained model replaces it.

