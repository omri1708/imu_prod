# docker/ws/Dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY server /app/server
RUN pip install --no-cache-dir websockets
EXPOSE 8766
CMD ["python","-m","server.stream_wfq_ws"]
