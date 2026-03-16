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

The response still includes per-image analysis details so the mobile app can inspect partial failures and image quality.
For the main product UI, prefer the top-level `analyzed_regions_summary`.

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

## Deploy on Render

This repo includes a ready-to-use [render.yaml](/c:/Users/Utilisateur/Desktop/Code/Weighty-App/module/render.yaml).

Render setup summary:

- Create a new Blueprint deployment from the repo
- Render will use:
  - `PYTHON_VERSION=3.11.11`
  - build command from `render.yaml`
  - start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
  - health check: `/health`

Important:

- The MediaPipe `.task` model is not committed because `models/` is ignored.
- `render.yaml` downloads the model automatically during the build.
- After deployment, the main URLs are:
  - `/docs`
  - `/test-ui`

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
  "analysis_feedback": "Good analysis base. Image quality looks solid overall. Reliable zones: Tete and Hanches. Weak or missing zones: Mollet / jambe gauche. The body fat estimate should still be treated as directional only.",
  "coaching_feedback": "Good overall composition. Mild abdominal fat retention is still likely. At an illustrative pace of about 0.5 kg per week, you could approach 24% estimated body fat around May 2026.",
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
