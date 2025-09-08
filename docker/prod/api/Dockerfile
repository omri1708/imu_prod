########## Builder ##########
FROM python:3.11-slim AS build
ENV PIP_NO_CACHE_DIR=1
WORKDIR /src
COPY . /src

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc curl ca-certificates tar \
    && rm -rf /var/lib/apt/lists/*

# (אופציונלי) אם אין k8s/ אך יש helm/ – הפקת מניפסט אחד לשימור עקבות K8s
RUN if [ ! -d k8s ] && [ -d helm ]; then \
      curl -sSL https://get.helm.sh/helm-v3.14.4-linux-amd64.tar.gz | tar -xz && \
      mv linux-amd64/helm /usr/local/bin/helm && rm -rf linux-amd64 ; \
      mkdir -p k8s ; \
      for d in helm/*; do \
        [ -f "$d/Chart.yaml" ] || continue ; \
        helm dependency update "$d" || true ; \
        helm template "$(basename "$d")" "$d" >> k8s/all.yaml || true ; \
      done ; \
    fi

# תלויות – צמידה ל‑Pydantic v1 ול‑FastAPI תואם (לא לשבור קוד קיים)
RUN python -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then \
       pip install --prefix=/install -r requirements.txt || true ; \
    fi && \
    pip install --prefix=/install \
       "fastapi<0.100" "pydantic<2.0" "uvicorn[standard]" websockets cryptography PyYAML \
       jinja2 httpx requests python-multipart orjson

# מכינים /mnt לעבודה ב‑nonroot (מונע PermissionError בזמן import)
RUN mkdir -p /tmp/mnt-root/data/imu_repo/var/demo-user && \
    touch /tmp/mnt-root/data/imu_repo/var/demo-user/.keep

########## Runtime ##########
FROM gcr.io/distroless/python3-debian12:nonroot
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app:/usr/local/lib/python3.11/site-packages
WORKDIR /app
COPY --from=build /install /usr/local
COPY --from=build --chown=nonroot:nonroot /tmp/mnt-root/ /mnt/
COPY --from=build /src /app
EXPOSE 8000
USER nonroot
# ENTRYPOINT של distroless הוא python → מריצים מודול:
CMD ["-m","uvicorn","server.http_api:APP","--host","0.0.0.0","--port","8000"]
