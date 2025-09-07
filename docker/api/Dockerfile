FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY ./server ./server
COPY ./policy ./policy
COPY ./provenance ./provenance
COPY ./adapters ./adapters
COPY ./ui ./ui
COPY ./helm ./helm
COPY ./k8s ./k8s
COPY ./requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn pydantic websockets cryptography pyyaml
EXPOSE 8000
CMD ["python","-m","uvicorn","server.http_api:APP","--host","0.0.0.0","--port","8000"]