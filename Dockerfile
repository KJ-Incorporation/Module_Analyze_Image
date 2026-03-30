FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEIGHTY_POSE_MODEL_PATH=/opt/weighty/models/pose_landmarker_lite.task

WORKDIR /opt/weighty

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY app ./app
COPY README.md ./README.md

RUN mkdir -p /opt/weighty/models && \
    curl -L https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task \
    -o /opt/weighty/models/pose_landmarker_lite.task

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
