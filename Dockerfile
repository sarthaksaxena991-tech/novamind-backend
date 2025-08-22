FROM python:3.9-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m venv /spleeter_env && /spleeter_env/bin/pip install --upgrade pip && /spleeter_env/bin/pip install spleeter==2.4.0
ENV SPLEETER_BIN=/spleeter_env/bin/spleeter REBUILD_INTERVAL_SECONDS=1800 PYTHONUNBUFFERED=1
COPY . .
CMD ["gunicorn","-w","1","-k","gthread","-t","900","app:app"]

