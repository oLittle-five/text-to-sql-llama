"""
RAG pipeline for text-to-SQL: retrieve similar training examples and build
a few-shot prompt for the base Llama-3 model (no fine-tuning).

Flow:
    1. Receive a test question + table schema
    2. Query ChromaDB for the top-k most similar training examples
    3. Assemble a few-shot prompt with those examples as demonstrations
    4. Return the prompt (caller handles LLM inference)

Usage (standalone test):
    python -m src.rag.rag_pipeline
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.data.prepare_dataset import format_prompt


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "wikisql_train"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """Retrieval-augmented few-shot prompt builder for text-to-SQL."""

    def __init__(self, db_dir: str = DEFAULT_DB_DIR, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k

        # Initialize embedding function (same model used at index time)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )

        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=db_dir)
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )
        print(f"RAG pipeline loaded: {self.collection.count():,} examples in index")

    def retrieve(self, question: str, columns: list[str], types: list[str],
                 top_k: int = None) -> list[dict]:
        """
        Retrieve the top-k most similar training examples.

        Args:
            question: Natural language question
            columns: Column names from the test table
            types: Column types from the test table
            top_k: Number of examples to retrieve (default: self.top_k)

        Returns:
            List of dicts with keys: prompt, completion, text, distance
        """
        k = top_k or self.top_k

        # Build the query string (same format as index documents)
        col_defs = ", ".join(
            f"{name} ({dtype})" for name, dtype in zip(columns, types)
        )
        query_text = f"Columns: {col_defs}\n\nQuestion: {question}"

        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query_text],
            n_results=k,
        )

        # Unpack results
        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "prompt": results["metadatas"][0][i]["prompt"],
                "completion": results["metadatas"][0][i]["completion"],
                "text": results["metadatas"][0][i]["text"],
                "distance": results["distances"][0][i],
            })

        return retrieved

    def build_few_shot_prompt(self, question: str, columns: list[str],
                               types: list[str], top_k: int = None) -> str:
        """
        Build a complete few-shot prompt with retrieved examples.

        The prompt format:
            Below are examples of converting questions to SQL queries.

            [Example 1]
            ### Input:
            Columns: ...
            Question: ...
            ### SQL:
            SELECT ...

            [Example 2]
            ...

            Now generate SQL for this question:

            ### Input:
            Columns: ...
            Question: ...
            ### SQL:

        Args:
            question: The test question
            columns: Column names from the test table
            types: Column types from the test table
            top_k: Number of few-shot examples

        Returns:
            Complete prompt string ready for LLM inference
        """
        examples = self.retrieve(question, columns, types, top_k)

        # Build the few-shot section
        parts = ["Below are examples of converting natural language questions to SQL queries.\n"]

        for i, ex in enumerate(examples, 1):
            # Each example is the full prompt + completion
            parts.append(f"[Example {i}]")
            parts.append(ex["text"])
            parts.append("")  # blank line separator

        parts.append("Now generate SQL for this question:\n")

        # Add the actual test input
        test_prompt = format_prompt(question, columns, types)
        parts.append(test_prompt)

        return "\n".join(parts)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test: retrieve examples for a sample question
    pipeline = RAGPipeline()

    # Example test case
    test_question = "How many people live in Tokyo?"
    test_columns = ["city", "country", "population"]
    test_types = ["text", "text", "real"]

    print("=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)
    examples = pipeline.retrieve(test_question, test_columns, test_types)
    for i, ex in enumerate(examples, 1):
        print(f"\n--- Retrieved Example {i} (distance: {ex['distance']:.4f}) ---")
        print(ex["text"][:200])

    print("\n" + "=" * 60)
    print("FULL FEW-SHOT PROMPT")
    print("=" * 60)
    prompt = pipeline.build_few_shot_prompt(test_question, test_columns, test_types)
    print(prompt)
