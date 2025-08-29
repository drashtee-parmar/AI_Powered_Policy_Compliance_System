# # --- Base image ---
# FROM python:3.12-slim

# # Faster, cleaner Python/pip behavior inside containers
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1

# # Avoid tz/locale prompts during apt operations
# ARG DEBIAN_FRONTEND=noninteractive

# WORKDIR /app

# # --- OS packages needed to build PyAudio and run Whisper ---
# # - build-essential/gcc/g++: compile C extensions (PyAudio wheel)
# # - portaudio19-dev (+ runtime libs): headers/libs required by PyAudio
# # - ffmpeg: for Whisper audio handling
# # - ca-certificates: TLS for pip/git
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       build-essential \
#       gcc \
#       g++ \
#       ffmpeg \
#       portaudio19-dev \
#       libasound2-dev \
#       libportaudio2 \
#       libportaudiocpp0 \
#       ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# # --- Python dependencies ---
# COPY requirements.txt .
# RUN python -m pip install --upgrade pip \
#  && pip install --no-cache-dir -r requirements.txt

# # --- App code ---
# COPY . .

# # --- Streamlit runtime ---
# ENV STREAMLIT_SERVER_HEADLESS=true \
#     STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
#     STREAMLIT_SERVER_PORT=8501
# EXPOSE 8501

# CMD ["streamlit", "run", "streamlit_chat.py"]


# --- Base ---
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# OS deps needed by ffmpeg/whisper etc. (adjust if not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY src/      /app/src/
COPY apps/     /app/apps/
COPY data/     /app/data/
COPY policies/ /app/policies/
# (optional) COPY .env /app/.env

EXPOSE 8501
CMD ["streamlit", "run", "apps/streamlit_app.py"]
