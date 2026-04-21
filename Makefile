.PHONY: data rag-index train eval serve dashboard lint format clean help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

data:  ## Prepare WikiSQL dataset (downloads + converts to JSONL)
	python -m src.data.prepare_dataset

rag-index:  ## Build ChromaDB vector index from training data
	python -m src.rag.build_index

train:  ## Fine-tune (open Colab notebook — requires GPU)
	@echo "Open notebooks/02_fine_tune_v1.ipynb on Google Colab (T4/A100)"

eval:  ## Run execution accuracy evaluation
	python -m src.eval.execution_accuracy

serve:  ## Start the FastAPI serving endpoint (requires GPU)
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

dashboard:  ## Launch the Streamlit comparison dashboard
	streamlit run src/dashboard/streamlit_app.py

lint:  ## Run linting with ruff
	ruff check src/ scripts/

format:  ## Auto-format code with ruff
	ruff format src/ scripts/

clean:  ## Remove generated files (data, chromadb, pycache)
	rm -rf data/processed/*.jsonl data/chroma_db/ __pycache__ src/**/__pycache__
