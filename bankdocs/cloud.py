"""Opt-in cloud adapters. Local tests use SDK stubs, never live cloud services."""
import io
import json
import os
from PIL import Image

from .core import MAX_BYTES


class TextractOCR:
    def __init__(self, client):
        self.client = client

    def extract(self, data, media_type):
        if media_type not in {"image/png", "image/jpeg"} or not 0 < len(data) <= MAX_BYTES:
            raise ValueError("Textract prototype accepts PNG/JPEG up to 5 MB")
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"PNG", "JPEG"} or image.width * image.height > 20_000_000:
                raise ValueError("Invalid image dimensions or format")
            image.verify()
        result = self.client.detect_document_text(Document={"Bytes": data})
        lines = [b["Text"] for b in result.get("Blocks", [])
                 if b.get("BlockType") == "LINE" and b.get("Text")]
        if not lines:
            raise ValueError("Textract returned no text")
        return ["\n".join(lines)], "aws-textract"


class BedrockGenerator:
    def __init__(self, client, model):
        if not model:
            raise ValueError("BEDROCK_MODEL_ID is required")
        self.client, self.model = client, model

    def complete(self, prompt):
        response = self.client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 800, "temperature": 0.0},
        )
        if response.get("stopReason") not in {"end_turn", "stop_sequence"}:
            raise ValueError("Model response was incomplete or blocked")
        text = "".join(b.get("text", "") for b in response["output"]["message"]["content"])
        if not text:
            raise ValueError("Model returned no text")
        return text


class Cortex:
    """Connection factory provides a fresh Snowflake connection per operation."""
    def __init__(self, connection_factory, embed_model="snowflake-arctic-embed-m", model=None):
        self.connection_factory = connection_factory
        self.embed_model, self.model = embed_model, model

    def _query(self, sql, values):
        conn = self.connection_factory()
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, values)
                row = cursor.fetchone()
                if not row:
                    raise ValueError("Cortex returned no row")
                return row[0]
            finally:
                cursor.close()
        finally:
            conn.close()

    def embed(self, texts):
        # Prototype is intentionally bounded; a production corpus needs batch indexing/cache.
        if not 1 <= len(texts) <= 200:
            raise ValueError("Embedding batch must contain 1-200 inputs")
        result = []
        for text in texts:
            value = self._query("SELECT AI_EMBED(%s, %s)::ARRAY", (self.embed_model, text))
            result.append(json.loads(value) if isinstance(value, str) else value)
        return result

    def complete(self, prompt):
        if not self.model:
            raise ValueError("CORTEX_MODEL_ID is required")
        return self._query(
            "SELECT AI_COMPLETE(%s, %s, {'temperature': 0, 'max_tokens': 800})",
            (self.model, prompt),
        )


def configured_adapters():
    """No SDK clients or credentials are loaded unless cloud mode is explicitly enabled."""
    mode = os.getenv("BANKDOCS_MODE", "local")
    if mode == "local":
        return None, None, None
    if mode != "cloud":
        raise ValueError("BANKDOCS_MODE must be local or cloud")
    import boto3
    from botocore.config import Config
    config = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2})
    region = os.environ["AWS_REGION"]
    ocr = TextractOCR(boto3.client("textract", region_name=region, config=config))
    backend = os.getenv("GENERATOR_BACKEND", "bedrock")
    embedder = None
    if os.getenv("RETRIEVER_BACKEND", "local") == "cortex" or backend == "cortex":
        import snowflake.connector
        def connect():
            return snowflake.connector.connect(
                account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
                private_key_file=os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"],
                role=os.environ["SNOWFLAKE_ROLE"], warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
                login_timeout=10, network_timeout=30,
            )
        cortex = Cortex(connect, model=os.getenv("CORTEX_MODEL_ID"))
        if os.getenv("RETRIEVER_BACKEND") == "cortex":
            embedder = cortex
    if backend == "bedrock":
        generator = BedrockGenerator(boto3.client("bedrock-runtime", region_name=region, config=config),
                                     os.getenv("BEDROCK_MODEL_ID"))
    elif backend == "cortex":
        generator = cortex
    else:
        raise ValueError("Unsupported generator backend")
    return ocr, generator, embedder
