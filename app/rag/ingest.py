import argparse
from pathlib import Path

from app.rag.ingestion import (
    IngestionPipeline
)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Markdown documents into the vector store."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Document paths. Defaults to all Markdown files in knowledge/.",
    )
    args = parser.parse_args()

    paths = [
        Path(path)
        for path in args.paths
    ] or sorted(Path("knowledge").glob("*.md"))

    if not paths:
        parser.error("No Markdown documents found in knowledge/.")

    pipeline = IngestionPipeline()

    for path in paths:
        print(f"Ingesting {path}...")
        pipeline.ingest_file(str(path))


if __name__ == "__main__":
    main()