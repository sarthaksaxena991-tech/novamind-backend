FROM python:3.9-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1200 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# upgrade pip first, then install with extra timeout/retries
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --retries 10 --timeout 1200 -r requirements.txt

COPY . .
ENV PORT=8000
CMD ["gunicorn","-w","2","-k","gthread","-b","0.0.0.0:8000","app:app","--timeout","300"]

