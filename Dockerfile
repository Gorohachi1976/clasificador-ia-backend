# Dockerfile optimizado para Google Cloud Run
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 600 app:app