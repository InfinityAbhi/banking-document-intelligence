FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-cloud.txt ./
RUN pip install --no-cache-dir -r requirements-cloud.txt && useradd -u 10001 -m appuser
COPY bankdocs/ ./bankdocs/
COPY deploy/logging.json ./logging.json
RUN mkdir -p /data && chown appuser:appuser /data
RUN chmod 1777 /tmp
VOLUME ["/tmp"]
USER 10001
ENV BANKDOCS_DB=/data/bankdocs.sqlite PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)"
CMD ["uvicorn", "bankdocs.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.json"]
