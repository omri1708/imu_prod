# syntax=docker/dockerfile:1.4

########## Builder ##########
FROM python:3.11-slim AS build
ENV PIP_NO_CACHE_DIR=1
WORKDIR /src
COPY . /src

# כלים מינימליים (wheels, helm אם תרצה לייצר k8s/all.yaml כשאין k8s/)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc curl ca-certificates tar \
    && rm -rf /var/lib/apt/lists/*

# אופציונלי: יצירת k8s/all.yaml מתוך helm אם אין k8s/
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

# הכנת ספריות פייתון לריצה (אין pip ב-runtime של distroless)
RUN python -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then \
       pip install --prefix=/install -r requirements.txt || true ; \
    fi && \
    pip install --prefix=/install \
       fastapi uvicorn[standard] pydantic websockets cryptography PyYAML \
       jinja2 httpx requests python-multipart orjson

# *** החלק שמונע את PermissionError ***
# מכינים תיקייה ריקה שתשוכפל לנתיב הסנדבוקס בריצה
RUN mkdir -p /tmp/imu_default_root && touch /tmp/imu_default_root/.keep

########## Runtime ##########
FROM gcr.io/distroless/python3-debian12:nonroot
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/usr/local/lib/python3.11/site-packages
WORKDIR /app

# ספריות פייתון שהותקנו בבילד
COPY --from=build /install /usr/local

# *** יצירת שרשרת /mnt/... עם הרשאות ל-nonroot (למנוע os.makedirs על /mnt) ***
COPY --from=build --chown=nonroot:nonroot /tmp/imu_default_root /mnt/data/imu_repo/var/demo-user

# קוד האפליקציה (כולל helm/ argocd/ וגם k8s/ אם יש/נוצר)
COPY --from=build /src /app

EXPOSE 8000
USER nonroot
# ב-distroless ה-ENTRYPOINT כבר python3.11 → מריצים מודול
CMD ["-m","uvicorn","server.http_api:APP","--host","0.0.0.0","--port","8000"]
