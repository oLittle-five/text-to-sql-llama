.PHONY: train eval serve dashboard lint

train:
	@echo "Launch notebooks/02_fine_tune_qlora.ipynb on Colab"

eval:
	python src/eval/execution_accuracy.py

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/dashboard/streamlit_app.py

lint:
	ruff check src/