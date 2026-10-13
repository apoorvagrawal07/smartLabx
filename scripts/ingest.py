from app.rag.ingestion import (
    IngestionPipeline
)


pipeline = IngestionPipeline()

pipeline.ingest_file(
    "knowledge/kafka.md"
)