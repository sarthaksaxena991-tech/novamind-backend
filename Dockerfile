FROM python:3.10-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8000
CMD ["gunicorn","-w","2","-k","gthread","-b","0.0.0.0:8000","app:app","--timeout","300"]