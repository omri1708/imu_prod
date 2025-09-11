# docker/prod/api/Dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY services/api /app
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir fastapi pydantic uvicorn[standard]
EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
