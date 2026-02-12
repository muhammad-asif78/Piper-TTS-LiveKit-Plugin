FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Minimal runtime deps for audio/media libs used by LiveKit stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY src /app/src

# Optional runtime overrides:
# - PIPER_PATH (default points to pip-installed CLI)
# - PIPER_MODEL_PATH (mount model into /models or change to your path)
ENV PIPER_PATH=/usr/local/bin/piper \
    PIPER_MODEL_PATH=/models/en_US-lessac-medium.onnx

WORKDIR /app/src

# Run the main voice agent in console mode by default.
CMD ["python", "agent.py", "console"]
