# ---------- Builder ----------
FROM python:3.11-slim AS build
ENV POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1
WORKDIR /src

# System deps for building wheels (minimized)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# App source
COPY ./server ./server
COPY ./policy ./policy
COPY ./provenance ./provenance
COPY ./adapters ./adapters
COPY ./helm ./helm
COPY ./k8s ./k8s
COPY ./ui ./ui
COPY ./requirements.txt ./requirements.txt

# Python deps (freeze wheels)
RUN python -m pip install --upgrade pip && \
    pip wheel --wheel-dir /wheels -r requirements.txt || true && \
    pip wheel --wheel-dir /wheels fastapi uvicorn pydantic websockets cryptography pyyaml

# ---------- Runtime ----------
FROM gcr.io/distroless/python3-debian12:nonroot
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=build /wheels /wheels
RUN ["python","-m","pip","install","--no-index","--find-links=/wheels","fastapi","uvicorn","pydantic","websockets","cryptography","pyyaml"]
COPY --from=build /src /app
EXPOSE 8000
USER nonroot
CMD ["python","-m","uvicorn","server.http_api:APP","--host","0.0.0.0","--port","8000"]
