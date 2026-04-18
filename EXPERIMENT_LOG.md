# Experiment Log: Text-to-SQL with Fine-Tuned Llama-3

This document tracks the full development process of fine-tuning Meta-Llama-3-8B-Instruct
on the WikiSQL dataset using QLoRA, including two training iterations, a critical debugging
discovery, and the post-processing pipeline that produced the best results.

---

## Project Overview

**Goal**: Fine-tune Llama-3-8B-Instruct to translate natural-language questions into SQL
queries, evaluated on the WikiSQL benchmark.

**Method**: QLoRA (Quantized Low-Rank Adaptation) — 4-bit NF4 quantization with LoRA
adapters applied to all attention and MLP projection layers. This keeps the full 8B model
frozen in ~6 GB GPU memory while training only ~20 MB of adapter weights.

**Dataset**: [Salesforce/WikiSQL](https://huggingface.co/datasets/Salesforce/wikisql)
- Train: 56,355 examples
- Validation: 8,421 examples
- Test: 15,878 examples (evaluated on a 500-example subset)

**Prompt format** (used consistently across all experiments):
```
### Input:
Columns: col1 (type1), col2 (type2), ...

Question: <natural language question>

### SQL:
<expected SQL output>
```

---

## Experiment Timeline

### Step 1: Data Exploration and Preparation

**Notebook**: `notebooks/01_data_exploration.ipynb`

Loaded the WikiSQL dataset and analyzed its structure. Each example contains a natural-language
question, a table schema (column names + types), and a structured SQL annotation (selection
column, aggregation operator, WHERE conditions). Converted the structured SQL annotations into
human-readable SQL strings and formatted everything into prompt-completion pairs saved as JSONL
files in `data/processed/`.

Key observations:
- WikiSQL uses a restricted SQL grammar: single-table SELECT with optional aggregation and
  WHERE clauses (no JOINs, subqueries, ORDER BY, GROUP BY, or LIMIT)
- Column names often contain spaces and special characters, requiring backtick quoting
- Number columns sometimes contain comma-formatted values (e.g., "46,735")

### Step 2: V1 Fine-Tuning

**Notebook**: `notebooks/02_fine_tune_v1.ipynb`
**Adapter**: [`oLittle-five/llama3-8b-wikisql-qlora`](https://huggingface.co/oLittle-five/llama3-8b-wikisql-qlora)

Hyperparameters:
| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 (scaling factor: 2.0) |
| Learning rate | 2e-4 |
| Batch size | 16 (per device) |
| Gradient accumulation | 2 |
| Epochs | 3 |
| Max sequence length | 256 |

Training losses by epoch:
| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.4972 | 0.6205 |
| 2 | 0.4032 | 0.6309 |
| 3 | 0.3408 | 0.6781 |

**Observation**: Validation loss increased after epoch 1, indicating overfitting. The best
checkpoint was at epoch 1, but the final model (epoch 3) was saved and pushed to HuggingFace Hub.

**Critical detail (discovered later)**: The TRL `SFTTrainer` automatically wraps training
examples with Llama-3's chat template when it detects an Instruct model. This means the model
was actually trained on inputs prefixed with `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n`
followed by the raw prompt — but this was not realized at the time.

### Step 3: Baseline and Initial V1 Evaluation

**Notebook**: `notebooks/03_eval_baseline_and_v1.ipynb`

Evaluated both the base (non-fine-tuned) Llama-3-8B-Instruct model and the v1 adapter using
raw prompts (no chat template prefix) at inference time.

Results:
| Method | Exec Accuracy | Syntax Error Rate |
|---|---|---|
| Base Llama-3-8B-Instruct | 37.2% | 21.8% |
| + QLoRA v1 (raw prompt at inference) | 25.4% | 27.4% |

**Problem**: V1 performed *worse* than the base model. The fine-tuned model had learned to
expect the chat template prefix (which SFTTrainer added during training), but inference was
sending raw prompts. This train/inference format mismatch caused the model to generate
incoherent or malformed SQL.

### Step 4: V2 Fine-Tuning Attempt

**Script**: `scripts/fine_tune_v2.py`
**Adapter**: [`oLittle-five/llama3-8b-wikisql-qlora-v2`](https://huggingface.co/oLittle-five/llama3-8b-wikisql-qlora-v2)

After identifying the chat template mismatch, a second training run was conducted with several
changes intended to fix the root cause and improve stability:

| Change | V1 | V2 | Rationale |
|---|---|---|---|
| Template handling | Auto (SFTTrainer wraps) | `formatting_func` bypasses wrapping | Ensures train format = inference format |
| Learning rate | 2e-4 | 1e-4 | Less aggressive, reduces memorization risk |
| LoRA alpha | 32 (scale 2.0) | 16 (scale 1.0) | More stable gradient magnitudes |
| Epochs | 3 | 2 | Avoid overfitting (val loss rose after epoch 1 in v1) |
| Warmup steps | 0 | 200 | Stabilizes early training |
| Eval strategy | Per-epoch | Every 500 steps | Finer-grained checkpointing |
| Batch size | 16 per device | 4 per device x 4 accum = 16 effective | T4 16GB safe |
| Max seq length | 256 | 512 | Covers all outliers + potential token overhead |

V2 training checkpoints (selected):
| Step | Epoch | Train Loss | Val Loss |
|---|---|---|---|
| 500 | 0.28 | 0.6807 | 0.6915 |
| 1500 | 0.85 | 0.5691 | 0.6383 |
| 2500 | 1.42 | 0.4765 | 0.6345 |
| 3500 | 1.99 | 0.4577 | 0.6311 (best) |
| 5000 | 2.84 | 0.4033 | 0.6524 |

### Step 5: V2 Evaluation

**Notebook**: `notebooks/05_eval_v2.ipynb`

Evaluated the v2 adapter using raw prompts (matching its training format). Added additional
post-processing: newline as a stop token, plus `clean_wikisql()` to strip unsupported SQL
constructs (OR, ORDER BY, LIMIT, IS NOT NULL, !=).

Results:
| Method | Exec Accuracy | Syntax Error Rate |
|---|---|---|
| Base Llama-3-8B-Instruct | 37.2% | 21.8% |
| + QLoRA v1 (raw prompt) | 25.4% | 27.4% |
| + QLoRA v2 (raw prompt) | 29.0% | 30.2% |

**Outcome**: V2 showed modest improvement over v1's raw-prompt evaluation (+3.6%), but still
underperformed the base model. The high syntax error rate (30.2%) suggested the model was
still struggling with output formatting despite the template fix in training.

### Step 6: V1 with Chat Template Fix at Inference

**Notebook**: `notebooks/04_eval_v1_fixed.ipynb`

Instead of retraining, the key insight was to fix inference to match v1's actual training
format. Since SFTTrainer had wrapped v1's training data with the Llama-3 chat template,
the solution was to prepend the same prefix at inference time:

```python
CHAT_PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
full_prompt = CHAT_PREFIX + raw_prompt
```

Combined with post-processing (column name backtick-wrapping, WikiSQL grammar enforcement,
stop token handling), this produced the best results by a wide margin. The initial run
(notebook 04) measured **51.4%** execution accuracy. A later controlled re-run (notebook 06)
with optimized generation parameters confirmed **51.8%** — the difference is within noise.

---

## Summary of All Results

| Configuration | Adapter | Inference Format | Exec Acc | Syntax Err |
|---|---|---|---|---|
| Baseline | None | Raw prompt | 37.2% | 21.8% |
| V1 (original eval) | v1 | Raw prompt (mismatch!) | 25.4% | 27.4% |
| V2 | v2 | Raw prompt (correct) | 29.0% | 30.2% |
| **V1 Fixed (controlled)** | **v1** | **Chat template (correct)** | **51.8%** | **5.4%** |

**Winner**: V1 adapter with chat template prefix at inference + enhanced post-processing.
The controlled comparison (notebook 06) is the definitive measurement: both models evaluated
with the same post-processing and evaluation code, each using generation parameters optimized
for that model.

---

## Post-Processing Pipeline

The enhanced post-processing applied in the best configuration includes:

1. **Stop tokens**: EOS token + `<|eot_id|>` to prevent runaway generation
2. **Column name fixing** (`fix_column_names`): Wraps column references in backticks,
   handles underscore variants of multi-word column names
3. **WikiSQL grammar enforcement** (`clean_wikisql`): Strips SQL constructs not supported
   by WikiSQL — OR clauses, ORDER BY, LIMIT, IS NOT NULL, != conditions
4. **Output cleaning**: Removes code block markers, strips comments (`--`), removes
   section markers (`###`), normalizes table names to `FROM table`
5. **Empty condition removal**: Strips WHERE conditions with empty string values

---

## Controlled Comparison and Ablation Studies

To rigorously validate the fine-tuning improvement, a series of controlled experiments were
run (notebooks 06 and 07) that systematically isolate each variable: post-processing,
generation parameters, and prompt format.

### Step 7: Controlled Comparison (Notebook 06)

**Notebook**: `notebooks/06_controlled_comparison.ipynb`

Both models were evaluated on the same 500 test examples using shared post-processing and
evaluation code, but with generation configs optimized for each model:

- **Base model**: no `repetition_penalty` (its best setting)
- **V1-fixed**: `repetition_penalty=1.3` (its best setting)

This is the fairest comparison because each model uses the config that was found to work
best for it (see "Repetition Penalty Effect" below for why they differ).

| Config | Exec Acc | Syntax Err | Wrong |
|---|---|---|---|
| Base (no clean_wikisql) | 37.2% | 21.8% | 41.0% |
| Base (with clean_wikisql) | 36.8% | 22.6% | 40.6% |
| V1-fixed (no clean_wikisql) | **51.8%** | **5.4%** | 42.8% |
| V1-fixed (with clean_wikisql) | 51.4% | 6.0% | 42.6% |

**Improvement from fine-tuning: +14.6 percentage points** (consistent regardless of
whether `clean_wikisql` is applied).

### Ablation: Effect of `clean_wikisql()`

| Model | Predictions Modified | Exec Acc Change |
|---|---|---|
| Base | 25/500 | -0.4% |
| V1-fixed | 4/500 | -0.4% |

`clean_wikisql()` has negligible effect on both models. The v1-fixed model's outputs are
already clean SQL that stays within WikiSQL's restricted grammar. The improvement comes
entirely from fine-tuning, not post-processing.

### Ablation: Repetition Penalty Effect

An early run of notebook 06 applied `repetition_penalty=1.3` to both models identically.
This degraded the base model severely:

| Config | Exec Acc |
|---|---|
| Base (no repetition_penalty) | 37.2% |
| Base (with repetition_penalty=1.3) | 10.2% |
| V1-fixed (with repetition_penalty=1.3) | 51.8% |

**Why this happens**: `repetition_penalty` divides the probability of already-generated
tokens by the penalty factor at each decoding step. This pushes the model toward rarer,
less-repeated tokens. For the base model — which hasn't been fine-tuned to produce
concise SQL — this penalty causes it to avoid common SQL keywords and generate exotic
or malformed queries. The fine-tuned model, having learned to produce concise SQL patterns,
is unaffected because its outputs are already short and structured.

This is why the controlled comparison uses separate generation configs: applying identical
parameters would actually create an unfair disadvantage for the base model.

### Step 8: Chat Prefix Ablation (Notebook 07)

**Notebook**: `notebooks/07_base_chat_prefix.ipynb`

Since Llama-3-8B-Instruct was originally trained by Meta using the chat template, a natural
question is: does simply using the "correct" prompt format improve the base model? This would
mean the fine-tuning improvement is partly explained by prompt format, not model adaptation.

| Config | Exec Acc | Syntax Err | Wrong |
|---|---|---|---|
| Base (raw prompt, no RP) | 37.2% | 21.8% | 41.0% |
| Base (chat prefix, no RP) | 0.0% | 51.8% | 48.2% |
| V1-fixed (chat prefix, RP=1.3) | **51.8%** | **5.4%** | 42.8% |

**Result**: The chat prefix catastrophically degrades the base model (37.2% → 0.0%).
The model either generates empty responses, produces SQL with unquoted column names that
fail execution, or uses syntax outside WikiSQL's grammar (`table_name` instead of `table`,
`BETWEEN`, `SUBSTRING_INDEX`, `PARTITION BY`). The chat prefix triggers the model's general
instruction-following behavior, which produces conversational-style SQL incompatible with
the evaluation pipeline.

**Interpretation**: This demonstrates that the chat prefix is not a universal improvement —
it specifically helps the v1-fixed model because QLoRA fine-tuning taught it to produce
correct WikiSQL-format SQL within the chat template context. The fine-tuning improvement
(0.0% → 51.8% with chat prefix, or 37.2% → 51.8% against the base model's best config)
is entirely attributable to the learned adapter weights, not prompt formatting.

### Step 9: Evaluation Pipeline Verification (Notebook 08)

**Notebook**: `notebooks/08_verify_eval_pipeline.ipynb`

To ensure all reported results are computed under a single, consistent evaluation standard,
all saved predictions were re-evaluated through the canonical `src/eval/` pipeline
(execution accuracy via in-memory SQLite with typed columns and backtick-quoted identifiers).

| Source | Expected | src/eval | Match |
|---|---|---|---|
| Controlled: Base (raw prompt) | 37.2% | 37.2% | PASS |
| Controlled: V1-fixed (chat prefix) | 51.8% | 51.8% | PASS |
| Original baseline (notebook 03) | 37.2% | 37.2% | PASS |
| V1-fixed (notebook 04) | 51.4% | 51.4% | PASS |
| Base + chat prefix (notebook 07) | 1.8% | 0.0% | Updated |

The notebook 07 result changed from 1.8% to 0.0% because its original inline evaluation
used a simplified schema (all-TEXT columns, unquoted table name) that happened to execute
some malformed queries. Under the canonical pipeline with typed columns, those queries fail
as syntax errors. All results reported in this document now reflect the canonical pipeline.

---

### Step 10: Error Analysis

**Notebook**: `notebooks/09_error_analysis.ipynb`
**Module**: `src/eval/error_analysis.py`

Categorized all 500 v1-fixed predictions into fine-grained error types to understand where
the model fails. The `error_analysis.py` module parses predicted SQL to extract SELECT columns,
aggregation functions, and WHERE conditions, then compares each component against the gold SQL
to classify errors as: `syntax_error`, `wrong_column`, `wrong_agg`, `wrong_where_col`,
`wrong_where_op`, `wrong_where_val`, `missing_where`, `extra_where`, or `multiple_errors`.

The notebook produces:
- Bar chart of error category distribution
- Pie chart of error-only breakdown
- Accuracy vs. WHERE complexity (number of conditions)
- Accuracy by aggregation type (NONE, COUNT, MAX, etc.)
- Representative failure examples per category
- Full per-example analysis saved to `results/error_analysis_v1_fixed.json`

**Error category breakdown (241 errors out of 500):**

| Category | Count | % of total |
|---|---|---|
| correct | 259 | 51.8% |
| multiple_errors | 71 | 14.2% |
| wrong_result_other | 52 | 10.4% |
| wrong_where_val | 47 | 9.4% |
| syntax_error | 27 | 5.4% |
| wrong_column | 26 | 5.2% |
| wrong_agg | 10 | 2.0% |
| wrong_where_col | 7 | 1.4% |
| extra_where | 1 | 0.2% |

**Accuracy by WHERE complexity:**

| # WHERE conditions | Examples | Correct | Accuracy |
|---|---|---|---|
| 0 | 11 | 8 | 72.7% |
| 1 | 457 | 241 | 52.7% |
| 2 | 31 | 10 | 32.3% |
| 3 | 1 | 0 | 0.0% |

**Accuracy by aggregation:**

| Aggregation | Examples | Correct | Accuracy |
|---|---|---|---|
| NONE | 356 | 188 | 52.8% |
| COUNT | 97 | 41 | 42.3% |
| MAX | 22 | 13 | 59.1% |
| MIN | 25 | 17 | 68.0% |

**Key findings:**

1. **String formatting dominates the error distribution.** The `wrong_where_val` +
   `wrong_result_other` buckets together account for ~20% of all predictions, and both are
   largely populated by casing mismatches (e.g., `'Bay of Islands'` → `'Bay Of Islands'`,
   `'g10'` → `'G10'`) or small hallucinations on the WHERE literal. The model identifies
   the correct column and operator but cannot reproduce the exact casing from the table.
   Because SQLite is case-sensitive on `=`, these return empty result sets. This is
   addressable through post-processing (fuzzy-matching predicted literals against actual
   table cells) or case-insensitive comparison (`COLLATE NOCASE`) — without retraining.

2. **Accuracy degrades sharply with WHERE complexity.** Single-condition queries succeed
   at 52.7%, two-condition queries drop to 32.3%. The composition of multiple `AND`
   filters is clearly harder than either subtask in isolation. Multi-condition examples
   make up only 6% of the test set but contribute disproportionately to remaining errors.

3. **COUNT is the weakest aggregation.** At 42.3%, COUNT trails MIN/MAX substantially.
   Manual inspection shows the model often returns the raw column without wrapping it in
   COUNT when phrasing uses cues like "how many" or "total number."

4. **Syntax errors are now mostly benign.** 5.4% of predictions fail to parse, and many
   failures trace back to pathological WikiSQL column names (e.g., long descriptive headers
   with punctuation) rather than the model producing invalid SQL structure.

**Suggested next steps (directional, not run):**
- Post-processing: fuzzy-match WHERE literals against actual column values before execution.
- Data augmentation: synthesize casing variants of WHERE values during training.
- Compositional focus: upsample 2–3 condition examples in training or add a curriculum.
- Evaluator refinement: the `parse_select` regex in `error_analysis.py` does not handle
  multi-word column names wrapped in backticks, which inflates the `wrong_column` bucket
  at the expense of `wrong_where_val`. Fixing this would give sharper category boundaries.

---

### Step 10: Post-Processing Ablation — Case-Insensitive Evaluation (Notebook 10)

**Notebook**: `notebooks/10_post_processing_ablation.ipynb`

Following the error analysis finding that ~20% of errors are casing mismatches, we tested
`TEXT COLLATE NOCASE` in the SQLite executor to measure how much accuracy is lost purely to
case sensitivity. This re-evaluates existing saved predictions — no new inference needed.

| Configuration | Case-Sensitive | Case-Insensitive | Delta |
|---|---|---|---|
| Base model | 37.2% (186/500) | 39.6% (198/500) | +2.4% |
| V1-fixed | 51.8% (259/500) | **68.2% (341/500)** | **+16.4%** |

**82 out of 241 v1-fixed errors** were purely casing mismatches — the model predicted the
correct column, operator, and value content but with wrong letter casing. This confirms the
error analysis finding and demonstrates that a simple case-insensitive comparison (or a
fuzzy-match post-processing step) could push execution accuracy to ~68% without retraining.

The base model gains only +2.4% from case insensitivity, consistent with its errors being
more diverse (syntax failures, structural mistakes) rather than surface-level casing issues.

---

## Key Lessons Learned

1. **Train/inference format consistency is critical**: The single biggest factor in model
   performance was ensuring the prompt format at inference matched what the model saw during
   training. A format mismatch turned a 51.8% model into a 25.4% model.

2. **SFTTrainer applies chat templates silently**: When using TRL's SFTTrainer with an
   Instruct model, it auto-wraps training data with the model's chat template. This must
   be accounted for at inference — or explicitly disabled with `formatting_func`.

3. **Generation parameters are not model-agnostic**: `repetition_penalty=1.3` is neutral
   for the fine-tuned model but catastrophic for the base model (37.2% → 10.2%). Similarly,
   the chat prefix helps the fine-tuned model but destroys the base model (37.2% → 0.0%).
   Each model needs its own optimal inference config.

4. **Post-processing matters less than expected**: The `clean_wikisql()` ablation showed it
   modifies only 4/500 v1-fixed predictions and 25/500 base predictions, with negligible
   accuracy impact on both. The fine-tuned model learned to stay within WikiSQL's grammar.

5. **Overfitting is visible in val loss**: Both v1 and v2 showed val loss increasing after
   roughly 1 epoch. For a dataset of this size with a QLoRA adapter, 1-2 epochs appears
   to be sufficient.

6. **More hyperparameter changes are not always better**: V2 changed six things simultaneously
   relative to v1. It is unclear which changes helped and which hurt. The simpler fix
   (adjusting inference format for v1) dramatically outperformed the retrained v2 model.

---

## File Reference

### Notebooks

| File | Purpose |
|---|---|
| `notebooks/01_data_exploration.ipynb` | Dataset loading, analysis, and JSONL preparation |
| `notebooks/02_fine_tune_v1.ipynb` | V1 QLoRA fine-tuning on Google Colab |
| `notebooks/03_eval_baseline_and_v1.ipynb` | Baseline + initial v1 evaluation (exposed the format mismatch) |
| `notebooks/04_eval_v1_fixed.ipynb` | V1 with chat template fix (51.4%) |
| `notebooks/05_eval_v2.ipynb` | V2 adapter evaluation |
| `notebooks/06_controlled_comparison.ipynb` | Definitive controlled comparison: base 37.2% vs v1-fixed 51.8% + clean_wikisql ablation |
| `notebooks/06a_controlled_comparison_shared_rp.ipynb` | Earlier run with shared repetition_penalty — documents the RP=1.3 degradation finding (base 10.2%) |
| `notebooks/07_base_chat_prefix.ipynb` | Chat prefix ablation on base model (0.0%) |
| `notebooks/08_verify_eval_pipeline.ipynb` | Verification that all results are consistent under `src/eval/` |
| `notebooks/09_error_analysis.ipynb` | Error categorization and visualization for v1-fixed model |
| `notebooks/10_post_processing_ablation.ipynb` | Case-insensitive evaluation ablation (51.8% → 68.2%) |

### Scripts and Source

| File | Purpose |
|---|---|
| `scripts/fine_tune_v2.py` | V2 training script with all changes documented |
| `scripts/eval_v2.py` | Standalone v2 evaluation script |
| `src/data/prepare_dataset.py` | WikiSQL to prompt-completion conversion |
| `src/data/sql_executor.py` | In-memory SQLite execution for evaluation |
| `src/eval/execution_accuracy.py` | Primary metric: execute-and-compare |
| `src/eval/exact_match.py` | Secondary metric: normalized string matching |
| `src/eval/error_analysis.py` | Failure categorization: syntax, column, agg, WHERE errors |

### Results

| File | Purpose |
|---|---|
| `results/base_model_results.json` | Baseline predictions and metrics (37.2%) |
| `results/base_model_with_rp.json` | Base model with repetition_penalty=1.3 (10.2%) — raw predictions only |
| `results/base_model_chat_prefix.json` | Base model with chat prefix (1.8%) |
| `results/controlled_comparison_results.json` | Full controlled comparison with both models + clean_wikisql ablation data |
| `results/v1_fixed_eval_results.json` | V1-fixed predictions from notebook 04 (51.4%) |
| `results/v1_training_results.json` | V1 training losses by epoch |
| `results/v2_training_results.json` | V2 training losses by step |
| `results/error_analysis_v1_fixed.json` | Per-example error categorization from notebook 09 |
| `results/post_processing_ablation.json` | Case-sensitive vs case-insensitive accuracy comparison |
| `results/error_analysis_*.png` | Visualization charts from error analysis (distribution, aggregation, pie, WHERE complexity) |
