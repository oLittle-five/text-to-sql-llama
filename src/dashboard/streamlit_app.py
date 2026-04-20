"""
Streamlit dashboard for comparing text-to-SQL approaches.

Run with:
    streamlit run src/dashboard/streamlit_app.py

Reads result JSON files from results/ to display:
    1. Summary metrics comparison (bar chart + table)
    2. Per-example prediction browser (side-by-side)
    3. Error breakdown analysis
"""

import json
import os
import sys

import streamlit as st
import pandas as pd

# ── Ensure project root is on path ────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.prepare_dataset import build_sql_string
from src.data.sql_executor import execute_sql


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Text-to-SQL: Model Comparison",
    page_icon="🔍",
    layout="wide",
)


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    """Load all result files and test examples."""
    results_dir = os.path.join(PROJECT_ROOT, "results")

    with open(os.path.join(results_dir, "base_model_results.json")) as f:
        base = json.load(f)
    with open(os.path.join(results_dir, "rag_results.json")) as f:
        rag = json.load(f)
    with open(os.path.join(results_dir, "v1_fixed_eval_results.json")) as f:
        finetuned = json.load(f)
    with open(os.path.join(results_dir, "controlled_comparison_results.json")) as f:
        controlled = json.load(f)
    with open(os.path.join(results_dir, "error_analysis_v1_fixed.json")) as f:
        error_analysis = json.load(f)
    with open(os.path.join(results_dir, "post_processing_ablation.json")) as f:
        ablation = json.load(f)

    return base, rag, finetuned, controlled, error_analysis, ablation


@st.cache_data
def load_test_examples():
    """Load WikiSQL test examples for the prediction browser."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikisql", trust_remote_code=True)
    return list(ds["test"].select(range(500)))


def normalize_result(result):
    """Normalize SQL result for comparison."""
    if result is None:
        return None
    return sorted([tuple(str(v).lower().strip() for v in row) for row in result])


def classify_prediction(pred_sql, example):
    """Classify a prediction as correct, syntax_error, or wrong_result."""
    table = example["table"]
    gold_sql = build_sql_string(example["sql"], table["header"], table["types"])

    pred_results, pred_error = execute_sql(table, pred_sql)
    gold_results, _ = execute_sql(table, gold_sql)

    if pred_error:
        return "syntax_error", gold_sql
    elif normalize_result(pred_results) == normalize_result(gold_results):
        return "correct", gold_sql
    else:
        return "wrong_result", gold_sql


# ── Load Data ─────────────────────────────────────────────────────────────────

base, rag, finetuned, controlled, error_analysis, ablation = load_results()

# Use controlled comparison numbers (definitive)
base_acc = controlled["base_model"]["without_clean_wikisql"]["exec_acc"]
base_syntax = controlled["base_model"]["without_clean_wikisql"]["syntax_err"]
ft_acc = controlled["v1_fixed"]["without_clean_wikisql"]["exec_acc"]
ft_syntax = controlled["v1_fixed"]["without_clean_wikisql"]["syntax_err"]
rag_acc = rag["execution_accuracy"]
rag_syntax = rag["syntax_error_rate"]


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.title("Text-to-SQL: Model Comparison Dashboard")
st.markdown(
    "Comparing three approaches to text-to-SQL on WikiSQL (500 test examples): "
    "**zero-shot base model**, **RAG few-shot retrieval**, and **QLoRA fine-tuning**."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Summary Metrics
# ══════════════════════════════════════════════════════════════════════════════

st.header("1. Summary Metrics")

# Metric cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Base Model (zero-shot)",
        f"{base_acc:.1%}",
        help="Llama-3-8B-Instruct with raw prompt, no examples",
    )
with col2:
    st.metric(
        "RAG (3-shot retrieval)",
        f"{rag_acc:.1%}",
        delta=f"+{rag_acc - base_acc:.1%} vs base",
        help="Base model + 3 retrieved few-shot examples from ChromaDB",
    )
with col3:
    st.metric(
        "Fine-tuned (QLoRA v1)",
        f"{ft_acc:.1%}",
        delta=f"+{ft_acc - base_acc:.1%} vs base",
        help="QLoRA fine-tuned with chat template prefix at inference",
    )

st.markdown("---")

# Comparison table
comparison_df = pd.DataFrame({
    "Approach": [
        "Base model (zero-shot)",
        "RAG 3-shot retrieval",
        "Fine-tuned v1 (QLoRA)",
        "Fine-tuned + COLLATE NOCASE",
    ],
    "Execution Accuracy": [
        f"{base_acc:.1%}",
        f"{rag_acc:.1%}",
        f"{ft_acc:.1%}",
        f"{ablation['v1_fixed']['case_insensitive']['execution_accuracy']:.1%}",
    ],
    "Syntax Error Rate": [
        f"{base_syntax:.1%}",
        f"{rag_syntax:.1%}",
        f"{ft_syntax:.1%}",
        f"{ft_syntax:.1%}",
    ],
    "Method": [
        "Raw prompt -> base Llama-3",
        "Retrieved examples -> base Llama-3",
        "Chat prefix -> fine-tuned Llama-3",
        "Chat prefix -> fine-tuned + case fix",
    ],
})

# Bar chart
chart_df = pd.DataFrame({
    "Approach": ["Base (zero-shot)", "RAG (3-shot)", "Fine-tuned (QLoRA)"],
    "Execution Accuracy": [base_acc, rag_acc, ft_acc],
    "Syntax Error Rate": [base_syntax, rag_syntax, ft_syntax],
})

col_chart, col_table = st.columns([1, 1])

with col_chart:
    st.subheader("Execution Accuracy")
    st.bar_chart(
        chart_df.set_index("Approach")["Execution Accuracy"],
        color="#4CAF50",
    )

with col_table:
    st.subheader("Full Comparison")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Syntax error comparison
st.subheader("Syntax Error Rate")
st.bar_chart(
    chart_df.set_index("Approach")["Syntax Error Rate"],
    color="#F44336",
)

st.markdown(
    "> **Key insight**: RAG improves accuracy (+6.8%) but barely reduces syntax errors "
    "(19.6% vs 21.8%). Fine-tuning both improves accuracy (+14.6%) *and* slashes syntax "
    "errors to 5.4%. The base model with RAG still generates SQL constructs outside "
    "WikiSQL's grammar (LIKE, DISTINCT, aliases)."
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Per-Example Prediction Browser
# ══════════════════════════════════════════════════════════════════════════════

st.header("2. Prediction Browser")
st.markdown("Compare predictions from all three approaches on individual examples.")

# Load test examples
try:
    test_examples = load_test_examples()
    has_test_data = True
except Exception as e:
    st.warning(f"Could not load WikiSQL test data: {e}. Prediction browser unavailable.")
    has_test_data = False

if has_test_data:
    # Get predictions
    base_preds = controlled["base_model"]["predictions"]
    rag_preds = rag["predictions"]
    ft_preds = controlled["v1_fixed"]["predictions"]

    # Filter controls
    filter_option = st.selectbox(
        "Filter examples",
        ["All", "All correct", "All wrong", "RAG beats base", "Fine-tuned beats RAG",
         "Syntax errors (any model)"],
    )

    # Classify all predictions
    @st.cache_data
    def classify_all():
        classifications = {"base": [], "rag": [], "finetuned": []}
        for i in range(500):
            ex = test_examples[i]
            classifications["base"].append(classify_prediction(base_preds[i], ex))
            classifications["rag"].append(classify_prediction(rag_preds[i], ex))
            classifications["finetuned"].append(classify_prediction(ft_preds[i], ex))
        return classifications

    classifications = classify_all()

    # Apply filter
    valid_indices = list(range(500))
    if filter_option == "All correct":
        valid_indices = [i for i in range(500)
                         if classifications["base"][i][0] == "correct"
                         and classifications["rag"][i][0] == "correct"
                         and classifications["finetuned"][i][0] == "correct"]
    elif filter_option == "All wrong":
        valid_indices = [i for i in range(500)
                         if classifications["base"][i][0] != "correct"
                         and classifications["rag"][i][0] != "correct"
                         and classifications["finetuned"][i][0] != "correct"]
    elif filter_option == "RAG beats base":
        valid_indices = [i for i in range(500)
                         if classifications["rag"][i][0] == "correct"
                         and classifications["base"][i][0] != "correct"]
    elif filter_option == "Fine-tuned beats RAG":
        valid_indices = [i for i in range(500)
                         if classifications["finetuned"][i][0] == "correct"
                         and classifications["rag"][i][0] != "correct"]
    elif filter_option == "Syntax errors (any model)":
        valid_indices = [i for i in range(500)
                         if any(classifications[m][i][0] == "syntax_error"
                                for m in ["base", "rag", "finetuned"])]

    st.caption(f"Showing {len(valid_indices)} examples matching filter.")

    if valid_indices:
        idx_in_filter = st.slider(
            "Example",
            0,
            len(valid_indices) - 1,
            0,
            format=f"Example %d of {len(valid_indices)}",
        )
        idx = valid_indices[idx_in_filter]

        ex = test_examples[idx]
        gold_sql = build_sql_string(ex["sql"], ex["table"]["header"], ex["table"]["types"])

        # Display question and schema
        st.subheader(f"Example {idx}")
        st.markdown(f"**Question:** {ex['question']}")
        col_info = ", ".join(
            f"`{name}` ({dtype})"
            for name, dtype in zip(ex["table"]["header"], ex["table"]["types"])
        )
        st.markdown(f"**Columns:** {col_info}")
        st.code(f"Gold SQL: {gold_sql}", language="sql")

        # Side-by-side predictions
        c1, c2, c3 = st.columns(3)

        for col, name, preds, model_key in [
            (c1, "Base (zero-shot)", base_preds, "base"),
            (c2, "RAG (3-shot)", rag_preds, "rag"),
            (c3, "Fine-tuned (QLoRA)", ft_preds, "finetuned"),
        ]:
            status, _ = classifications[model_key][idx]
            icon = {"correct": "✅", "syntax_error": "⚠️", "wrong_result": "❌"}[status]
            with col:
                st.markdown(f"**{name}** {icon}")
                st.code(preds[idx], language="sql")
                st.caption(status.replace("_", " ").title())
    else:
        st.info("No examples match this filter.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Error Analysis (Fine-tuned model)
# ══════════════════════════════════════════════════════════════════════════════

st.header("3. Error Analysis (Fine-tuned Model)")
st.markdown(
    "Breakdown of the 241 errors from the fine-tuned v1 model (51.8% accuracy). "
    "Most errors are surface-form issues, not semantic failures."
)

# Error type breakdown
summary = error_analysis["summary"]
error_types = {k: v for k, v in summary.items() if k != "correct"}

err_df = pd.DataFrame({
    "Error Type": [
        "Wrong WHERE value",
        "Wrong result (other)",
        "Multiple errors",
        "Syntax error",
        "Wrong column",
        "Wrong aggregation",
        "Wrong WHERE column",
        "Extra WHERE clause",
    ],
    "Count": [
        error_types.get("wrong_where_val", 0),
        error_types.get("wrong_result_other", 0),
        error_types.get("multiple_errors", 0),
        error_types.get("syntax_error", 0),
        error_types.get("wrong_column", 0),
        error_types.get("wrong_agg", 0),
        error_types.get("wrong_where_col", 0),
        error_types.get("extra_where", 0),
    ],
})
err_df = err_df.sort_values("Count", ascending=False)

col_err_chart, col_err_detail = st.columns([1, 1])

with col_err_chart:
    st.subheader("Error Distribution")
    st.bar_chart(err_df.set_index("Error Type")["Count"], color="#FF9800")

with col_err_detail:
    st.subheader("Accuracy by Query Complexity")

    # WHERE condition complexity
    where_data = error_analysis["accuracy_by_where_conditions"]
    where_df = pd.DataFrame({
        "WHERE Conditions": ["0", "1", "2", "3"],
        "Accuracy": [
            f"{where_data.get('0', {}).get('correct', 0) / max(where_data.get('0', {}).get('total', 1), 1):.1%}",
            f"{where_data.get('1', {}).get('correct', 0) / max(where_data.get('1', {}).get('total', 1), 1):.1%}",
            f"{where_data.get('2', {}).get('correct', 0) / max(where_data.get('2', {}).get('total', 1), 1):.1%}",
            f"{where_data.get('3', {}).get('correct', 0) / max(where_data.get('3', {}).get('total', 1), 1):.1%}",
        ],
        "Total Examples": [
            where_data.get("0", {}).get("total", 0),
            where_data.get("1", {}).get("total", 0),
            where_data.get("2", {}).get("total", 0),
            where_data.get("3", {}).get("total", 0),
        ],
    })
    st.dataframe(where_df, use_container_width=True, hide_index=True)

    # Aggregation accuracy
    agg_data = error_analysis["accuracy_by_aggregation"]
    agg_rows = []
    for agg_name in ["NONE", "COUNT", "MAX", "MIN"]:
        if agg_name in agg_data:
            vals = agg_data[agg_name]
            acc = vals["correct"] / max(vals["total"], 1)
            agg_rows.append({
                "Aggregation": agg_name if agg_name != "NONE" else "No aggregation",
                "Accuracy": f"{acc:.1%}",
                "Correct": vals["correct"],
                "Total": vals["total"],
            })
    agg_df = pd.DataFrame(agg_rows)
    st.dataframe(agg_df, use_container_width=True, hide_index=True)

# Case sensitivity finding
st.subheader("Case Sensitivity Impact")
col_cs, col_ci = st.columns(2)
with col_cs:
    st.metric(
        "Case-Sensitive (standard)",
        f"{ablation['v1_fixed']['case_sensitive']['execution_accuracy']:.1%}",
    )
with col_ci:
    st.metric(
        "Case-Insensitive (COLLATE NOCASE)",
        f"{ablation['v1_fixed']['case_insensitive']['execution_accuracy']:.1%}",
        delta=f"+{ablation['v1_fixed']['case_insensitive']['execution_accuracy'] - ablation['v1_fixed']['case_sensitive']['execution_accuracy']:.1%}",
    )
st.markdown(
    f"**{ablation['v1_fixed']['flipped_wrong_to_correct']}** of 241 errors are purely casing mismatches. "
    "A simple case-insensitive comparison or fuzzy-match post-processing could recover these "
    "without retraining."
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Methodology Notes
# ══════════════════════════════════════════════════════════════════════════════

st.header("4. Methodology")
st.markdown("""
**Dataset**: [WikiSQL](https://github.com/salesforce/WikiSQL) — 56K train / 8K val / 15K test.
Evaluated on first 500 test examples.

**Model**: [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct),
loaded in 4-bit (NF4) quantization.

**Approaches**:

- **Base (zero-shot)**: Raw prompt with column definitions and question. No examples, no fine-tuning.
  Generation: `max_new_tokens=128`, `do_sample=False`, no repetition penalty.

- **RAG (3-shot)**: Retrieve top-3 similar training examples from ChromaDB
  (embedded with `all-MiniLM-L6-v2`), prepend as few-shot demonstrations.
  Generation: same as base but with `repetition_penalty=1.2` (required to prevent output looping
  with longer prompts).

- **Fine-tuned (QLoRA v1)**: LoRA adapters (r=16, alpha=32) trained for 3 epochs with SFTTrainer.
  Inference uses chat template prefix to match training format.
  Generation: `repetition_penalty=1.3`.

**Evaluation**: Execution accuracy — predicted SQL is run against an in-memory SQLite database
built from the WikiSQL table, and results are compared to the gold SQL output.

**Adapters**: [oLittle-five/llama3-8b-wikisql-qlora](https://huggingface.co/oLittle-five/llama3-8b-wikisql-qlora)

For full experiment history, see [EXPERIMENT_LOG.md](https://github.com/oLittle-five/text-to-sql-llama/blob/main/EXPERIMENT_LOG.md).
""")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Text-to-SQL Fine-Tuning Project")
