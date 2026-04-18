"""
Build a ChromaDB vector index from WikiSQL training examples.

Each training example is embedded as: "Question: {question} | Columns: {col1}, {col2}, ..."
so that at retrieval time, similar (question, schema) pairs surface as few-shot examples.

Usage:
    python -m src.rag.build_index                     # default: data/processed/train.jsonl → data/chroma_db/
    python -m src.rag.build_index --input train.jsonl --output ./my_db
"""

import argparse
import json
import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_INPUT = "data/processed/train.jsonl"
DEFAULT_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "wikisql_train"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~80 MB, runs on CPU
BATCH_SIZE = 512  # ChromaDB add batch size


def load_training_examples(jsonl_path: str) -> list[dict]:
    """Load prompt-completion pairs from JSONL produced by prepare_dataset.py."""
    examples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def build_document(example: dict) -> str:
    """
    Build the text string that gets embedded.

    We embed the *input side* only (question + column names) so that retrieval
    finds training examples whose schemas and questions resemble the test query.
    The SQL completion is stored as metadata for later use in the few-shot prompt.
    """
    # Extract question and column info from the prompt
    # Prompt format: "### Input:\nColumns: ...\n\nQuestion: ...\n\n### SQL:\n"
    prompt = example["prompt"]
    # Use everything between "### Input:\n" and "\n\n### SQL:\n"
    input_section = prompt.split("### Input:\n")[-1].split("\n\n### SQL:")[0].strip()
    return input_section


def build_index(input_path: str, db_dir: str) -> None:
    """Embed all training examples and store in ChromaDB."""

    print(f"Loading training examples from {input_path} ...")
    examples = load_training_examples(input_path)
    print(f"  Loaded {len(examples):,} examples")

    # Set up embedding function
    print(f"Initializing embedding model: {EMBEDDING_MODEL} ...")
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    # Set up ChromaDB (persistent on disk)
    os.makedirs(db_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=db_dir)

    # Delete existing collection if present (for idempotent rebuilds)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # collection doesn't exist yet — that's fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    # Prepare documents, metadata, and IDs
    print("Building documents and embedding ...")
    documents = []
    metadatas = []
    ids = []

    for i, ex in enumerate(examples):
        doc = build_document(ex)
        documents.append(doc)
        metadatas.append({
            "prompt": ex["prompt"],
            "completion": ex["completion"],
            "text": ex["text"],  # full prompt + completion (for few-shot)
        })
        ids.append(f"train_{i}")

    # Add in batches (ChromaDB has per-call limits)
    total = len(documents)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        if (start // BATCH_SIZE) % 20 == 0 or end == total:
            print(f"  Added {end:,} / {total:,} examples")

    print(f"\nDone! ChromaDB index saved to {db_dir}/")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Documents:  {collection.count():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB index from WikiSQL training data")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to training JSONL")
    parser.add_argument("--output", default=DEFAULT_DB_DIR, help="Path to ChromaDB directory")
    args = parser.parse_args()

    build_index(args.input, args.output)
