FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY ./server ./server
COPY ./server/stream_wfq.py ./server/stream_wfq.py
COPY ./server/stream_wfq_ws.py ./server/stream_wfq_ws.py
RUN python -m pip install websockets
EXPOSE 8766
CMD ["python","server/stream_wfq_ws.py"]