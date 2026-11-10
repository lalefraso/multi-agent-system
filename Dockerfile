FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY getnet_ingestion/requirements.txt ./getnet_ingestion/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY getnet_ingestion ./getnet_ingestion
COPY settings.toml ./settings.toml

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
