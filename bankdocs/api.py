import hmac
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pypdf.errors import PyPdfError

from .cloud import configured_adapters
from .core import MAX_BYTES, Store, answer, extract_local, search

DEMO_KEY = "synthetic-local-demo"
LOG = logging.getLogger("bankdocs.requests")


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class Records(BaseModel):
    rows: list[dict] = Field(min_length=1, max_length=1000)


def create_app(db_path=None, api_key=None, adapters=None):
    key = api_key or os.getenv("BANKDOCS_API_KEY", DEMO_KEY)
    if os.getenv("BANKDOCS_MODE", "local") != "local" and (key == DEMO_KEY or len(key) < 24):
        raise ValueError("Cloud mode requires an explicit API key of at least 24 characters")
    store = Store(db_path or os.getenv("BANKDOCS_DB", ".runtime/bankdocs.sqlite"))
    ocr, generator, embedder = adapters if adapters is not None else configured_adapters()
    app = FastAPI(title="Banking Document Workbench", version="1.0.0")
    app.state.store = store

    def authorize(x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key.encode(), key.encode()):
            raise HTTPException(401, "API key required")

    @app.middleware("http")
    async def observe(request, call_next):
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        # Restrict total request body, including requests sent without Content-Length.
        received = 0
        original_receive = request._receive
        async def bounded_receive():
            nonlocal received
            message = await original_receive()
            received += len(message.get("body", b""))
            if received > MAX_BYTES + 20_000:
                raise HTTPException(413, "Request exceeds 5 MB limit")
            return message
        request._receive = bounded_receive
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        LOG.info(json.dumps({"request_id": request_id, "method": request.method,
                            "path": request.url.path, "status": response.status_code,
                            "elapsed_ms": round((time.perf_counter() - start) * 1000, 2)}))
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "ocr": "aws-textract" if ocr else "local-tesseract",
                "generation": "cloud" if generator else "local-extractive",
                "retrieval": "cortex" if embedder else "tfidf"}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "static/index.html")

    @app.get("/static/{filename}", include_in_schema=False)
    def static(filename: str):
        if filename not in {"app.js", "style.css"}:
            raise HTTPException(404)
        return FileResponse(Path(__file__).parent / "static" / filename)

    @app.get("/api/stats", dependencies=[Depends(authorize)])
    def stats():
        return store.stats()

    @app.get("/api/documents", dependencies=[Depends(authorize)])
    def documents():
        return store.list_documents()

    @app.get("/api/issues", dependencies=[Depends(authorize)])
    def issues():
        return store.issues()

    @app.post("/api/documents", dependencies=[Depends(authorize)])
    def upload(doc_id: str = Form(...), title: str = Form(...), file: UploadFile = File(...)):
        data = file.file.read(MAX_BYTES + 1)
        try:
            if len(data) > MAX_BYTES:
                raise ValueError("Document exceeds 5 MB")
            pages, extractor = ocr.extract(data, file.content_type) if ocr else extract_local(data, file.content_type)
            return store.ingest(doc_id, title, pages, data, extractor)
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            raise HTTPException(503, "Local OCR is unavailable or timed out") from exc
        except (ValueError, OSError, PyPdfError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/records", dependencies=[Depends(authorize)])
    def records(body: Records):
        try:
            return store.ingest_records(body.rows)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/search", dependencies=[Depends(authorize)])
    def retrieval(body: Query):
        try:
            return search(store, body.question, embedder=embedder)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/ask", dependencies=[Depends(authorize)])
    def ask(body: Query):
        try:
            return answer(store, body.question, generator, embedder)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, "No validated answer could be produced") from exc

    return app
