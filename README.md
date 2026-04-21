# Text-to-SQL with Fine-Tuned Llama-3

Fine-tuned [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
on the [WikiSQL](https://github.com/salesforce/WikiSQL) benchmark using QLoRA (Quantized
Low-Rank Adaptation), achieving **51.8% execution accuracy** on a 500-example test subset
with only ~20 MB of trainable adapter weights.

## Highlights

- **QLoRA fine-tuning** with 4-bit NF4 quantization — trains on a single T4 16 GB GPU
- **51.8% execution accuracy** (up from 37.2% baseline), reaching **68.2%** with case-insensitive evaluation; syntax errors reduced from 21.8% to 5.4%
- **RAG baseline comparison**: retrieval-augmented few-shot prompting achieves 44.0% — fine-tuning outperforms RAG by +7.8%
- **Controlled ablation studies** isolating the effects of post-processing, generation parameters, and prompt format
- Documented a critical finding about **train/inference format consistency** with TRL's SFTTrainer and Llama-3's chat template

## Results

| Configuration | Exec Accuracy | Syntax Error Rate |
|---|---|---|
| Base Llama-3-8B-Instruct (no fine-tuning) | 37.2% | 21.8% |
| + RAG 3-shot retrieval (no fine-tuning) | 44.0% | 19.6% |
| + QLoRA v1 (wrong inference format) | 25.4% | 27.4% |
| + QLoRA v2 (retrained with format fix) | 29.0% | 30.2% |
| **+ QLoRA v1 (chat template fix at inference)** | **51.8%** | **5.4%** |
| + Case-insensitive evaluation (COLLATE NOCASE) | **68.2%** | 5.4% |

A controlled comparison (notebook 06) confirms the improvement comes from fine-tuning, not
post-processing or generation parameters. The `clean_wikisql()` post-processing modifies
only **4 out of 500** v1-fixed predictions with negligible accuracy impact (-0.4%). Additional
ablations show that `repetition_penalty` and the chat prefix both hurt the base model but
help (or are neutral for) the fine-tuned model — each model was evaluated with its optimal
config. All reported results have been verified against a single canonical evaluation pipeline
(`src/eval/`, notebook 08). See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for the full
experiment history and analysis.

### Error Analysis (notebook 09)

Categorizing the 241 remaining errors reveals that the model has learned the task structure
well — most failures are surface-form issues rather than semantic mistakes:

- **~20% of predictions fail on string casing/formatting** in WHERE values (e.g., `'Bay of Islands'` → `'Bay Of Islands'`). The model picks the right column and operator but cannot reproduce the exact casing from the table; SQLite's case-sensitive `=` then returns empty results.
- **Multi-condition queries are disproportionately hard**: 52.7% accuracy on 1-condition queries drops to 32.3% on 2-condition queries.
- **COUNT (42.3%) is the weakest aggregation**, trailing MIN (68.0%) and MAX (59.1%).
- **Syntax errors are down to 5.4%**, and most trace to pathological WikiSQL column names rather than malformed model output.

A post-processing ablation (notebook 10) confirms this: adding `TEXT COLLATE NOCASE` to the
SQLite executor recovers 82 of the 241 remaining errors, pushing v1-fixed from 51.8% to
68.2% — purely by resolving casing mismatches, with no retraining needed.

These findings suggest the highest-leverage improvements are post-processing (fuzzy-matching
WHERE literals to actual table cells) and targeted data augmentation for multi-condition
and COUNT queries — not a larger model.

## Project Structure

```
text-to-sql-llama/
├── EXPERIMENT_LOG.md                 # Detailed experiment log and analysis
├── notebooks/
│   ├── 01_data_exploration.ipynb         # Dataset analysis and JSONL preparation
│   ├── 02_fine_tune_v1.ipynb             # V1 QLoRA training (Google Colab)
│   ├── 03_eval_baseline_and_v1.ipynb     # Baseline + v1 evaluation (exposed format mismatch)
│   ├── 04_eval_v1_fixed.ipynb            # V1 with chat template fix (51.4%)
│   ├── 05_eval_v2.ipynb                  # V2 adapter evaluation
│   ├── 06_controlled_comparison.ipynb    # Controlled comparison: base 37.2% vs v1-fixed 51.8%
│   ├── 06a_controlled_comparison_shared_rp.ipynb  # Earlier run documenting repetition_penalty effect
│   ├── 07_base_chat_prefix.ipynb         # Chat prefix ablation on base model (0.0%)
│   ├── 08_verify_eval_pipeline.ipynb    # Verification: all results consistent under src/eval/
│   ├── 09_error_analysis.ipynb          # Error categorization and visualization
│   ├── 10_post_processing_ablation.ipynb  # Case-insensitive evaluation ablation
│   ├── 11_rag_baseline.ipynb            # RAG few-shot baseline (44.0%)
│   └── 12_serving_demo.ipynb           # FastAPI serving demo (Colab)
├── scripts/
│   ├── fine_tune_v2.py                   # V2 training script (documented changes)
│   ├── eval_v2.py                        # Standalone V2 evaluation script
│   └── test_api.py                       # Automated API endpoint tests
├── results/                              # Predictions and metrics from each experiment
├── models/
│   └── v2_adapter/                       # V2 LoRA adapter weights (local)
├── src/
│   ├── data/
│   │   ├── prepare_dataset.py            # WikiSQL → prompt-completion conversion
│   │   └── sql_executor.py              # In-memory SQLite execution engine
│   ├── eval/
│   │   ├── execution_accuracy.py         # Primary metric: execute and compare
│   │   ├── exact_match.py               # Secondary metric: string matching
│   │   └── error_analysis.py            # Failure categorization by error type
│   ├── rag/
│   │   ├── build_index.py               # Embed training examples into ChromaDB
│   │   └── rag_pipeline.py              # Retrieve similar examples → few-shot prompt
│   ├── serving/
│   │   ├── app.py                       # FastAPI endpoint (/predict, /predict/batch, /health)
│   │   └── Dockerfile                   # GPU container for deployment
│   └── dashboard/
│       └── streamlit_app.py             # Side-by-side comparison UI
├── data/
│   └── processed/                        # Train/val/test JSONL splits
├── configs/
│   └── training_config.yaml              # Centralized hyperparameters
├── Makefile                              # make data, make serve, make dashboard, etc.
└── .github/workflows/lint.yml            # CI: ruff lint + format check
```

## Quick Start

```bash
pip install -r requirements.txt
make help                          # show all available commands
```

## Usage

**Data preparation** (local, no GPU):
```bash
make data                          # download WikiSQL + convert to JSONL
```

**Fine-tuning** (requires GPU — designed for Google Colab):
Open `notebooks/02_fine_tune_v1.ipynb` on Colab with a T4 or A100 runtime.

**Evaluation** (requires GPU):
Open `notebooks/06_controlled_comparison.ipynb` for the definitive comparison,
or `notebooks/04_eval_v1_fixed.ipynb` for the best fine-tuned result.

**RAG baseline** (index locally, evaluate on Colab):
```bash
make rag-index                     # build ChromaDB vector index (~5 min, CPU)
```
Then open `notebooks/11_rag_baseline.ipynb` on Colab for the full evaluation.

**API serving** (requires GPU):
```bash
make serve                         # start FastAPI at http://localhost:8000
```
Or see `notebooks/12_serving_demo.ipynb` to run on Colab. API docs at `/docs`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "How many people live in Tokyo?",
       "columns": ["city", "country", "population"],
       "types": ["text", "text", "real"]}'
```

**Dashboard** (local, no GPU):
```bash
make dashboard                     # open Streamlit at http://localhost:8501
```

## Trained Adapters (HuggingFace Hub)

- **V1**: [`oLittle-five/llama3-8b-wikisql-qlora`](https://huggingface.co/oLittle-five/llama3-8b-wikisql-qlora) — use with chat template prefix at inference
- **V2**: [`oLittle-five/llama3-8b-wikisql-qlora-v2`](https://huggingface.co/oLittle-five/llama3-8b-wikisql-qlora-v2) — use with raw prompts

## Key Technical Finding

TRL's `SFTTrainer` silently wraps training data with the model's chat template when using
Instruct models. If inference uses raw prompts (without the chat template), the format mismatch
can degrade accuracy by over 25 percentage points. The fix is either:

1. **At inference**: Prepend the chat template prefix to match training format (used for best result)
2. **At training**: Pass a `formatting_func` to SFTTrainer to bypass auto-wrapping (used in v2)

This finding is documented in detail in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## License

This project is for educational and research purposes.
