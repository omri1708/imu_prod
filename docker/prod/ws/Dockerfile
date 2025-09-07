# ---------- Builder ----------
FROM python:3.11-slim AS build
WORKDIR /src
RUN python -m pip install --upgrade pip && pip wheel --wheel-dir /wheels websockets

# ---------- Runtime ----------
FROM gcr.io/distroless/python3-debian12:nonroot
WORKDIR /app
COPY --from=build /wheels /wheels
RUN ["python","-m","pip","install","--no-index","--find-links=/wheels","websockets"]
COPY ./server/stream_wfq.py ./server/stream_wfq.py
COPY ./server/stream_wfq_ws.py ./server/stream_wfq_ws.py
EXPOSE 8766
USER nonroot
CMD ["python","server/stream_wfq_ws.py"]
