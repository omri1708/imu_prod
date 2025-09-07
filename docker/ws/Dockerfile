FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app
WORKDIR /app
COPY ./server ./server
RUN python -m pip install websockets
EXPOSE 8766
CMD ["python","-m","server.stream_wfq_ws"]
