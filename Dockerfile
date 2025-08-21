# ---- Base ----
FROM python:3.9-slim

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 gcc && \
    rm -rf /var/lib/apt/lists/*

# Runtime env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Workdir
WORKDIR /app

# Copy reqs first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port
EXPOSE 8080

# Start with Gunicorn
# app:app => <file>:<Flask instance>
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-t", "180", "-b", "0.0.0.0:8080", "app:app"]
