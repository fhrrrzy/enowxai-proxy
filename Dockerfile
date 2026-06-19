FROM python:3.12-slim
WORKDIR /app
# Cache bust: 20260619-v5
RUN pip install --no-cache-dir fastapi uvicorn httpx
COPY proxy.py .
CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "3010"]
