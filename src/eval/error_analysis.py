"""
Error analysis for Text-to-SQL predictions.

Classifies prediction errors into categories:
  - correct: execution result matches gold
  - syntax_error: SQL fails to parse/execute
  - wrong_column: correct agg but wrong SELECT column
  - wrong_agg: correct column but wrong aggregation
  - wrong_where_col: WHERE references wrong column(s)
  - wrong_where_op: WHERE uses wrong operator(s)
  - wrong_where_val: WHERE uses wrong value(s)
  - missing_where: prediction is missing WHERE clause(s) present in gold
  - extra_where: prediction has extra WHERE clause(s) not in gold
  - multiple_errors: more than one of the above
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.sql_executor import execute_sql
from src.data.prepare_dataset import build_sql_string, AGG_OPS


def parse_select(sql: str) -> tuple[str, str]:
    """
    Extract (aggregation, column) from a SELECT clause.

    Returns:
        (agg, col) where agg is '' for no aggregation, or 'MAX', 'MIN', etc.
    """
    sql = sql.strip()
    # Match SELECT [AGG(]col[)] FROM ...
    m = re.match(
        r"SELECT\s+(MAX|MIN|COUNT|SUM|AVG)\s*\(\s*`?([^`\)]+)`?\s*\)",
        sql,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper(), m.group(2).strip()

    # No aggregation — prefer backtick-quoted column (may contain spaces)
    m = re.match(r"SELECT\s+`([^`]+)`\s+FROM", sql, re.IGNORECASE)
    if m:
        return "", m.group(1).strip()

    # Fallback: unquoted single-token column
    m = re.match(r"SELECT\s+([^\s,]+)\s+FROM", sql, re.IGNORECASE)
    if m:
        return "", m.group(1).strip()

    return "", ""


def parse_where(sql: str) -> list[dict]:
    """
    Extract WHERE conditions from a SQL string.

    Returns:
        List of {'col': str, 'op': str, 'val': str} dicts.
    """
    # Find everything after WHERE
    m = re.search(r"WHERE\s+(.+)$", sql, re.IGNORECASE)
    if not m:
        return []

    where_str = m.group(1).strip()
    conditions = []

    # Split on AND (case-insensitive)
    parts = re.split(r"\s+AND\s+", where_str, flags=re.IGNORECASE)

    for part in parts:
        part = part.strip()
        # Match: `col` op 'val' or `col` op number
        cm = re.match(
            r"`?([^`]+)`?\s*(=|>|<|>=|<=|!=)\s*(.+)$",
            part,
        )
        if cm:
            col = cm.group(1).strip()
            op = cm.group(2).strip()
            val = cm.group(3).strip().strip("'\"")
            conditions.append({"col": col, "op": op, "val": val})

    return conditions


def classify_error(
    predicted_sql: str,
    gold_sql_dict: dict,
    table: dict,
) -> dict:
    """
    Classify a single prediction error.

    Args:
        predicted_sql: SQL string from the model
        gold_sql_dict: WikiSQL structured sql dict (sel, agg, conds)
        table: WikiSQL table dict

    Returns:
        dict with keys: category, details, predicted_sql, gold_sql
    """
    headers = table["header"]
    types = table["types"]
    gold_sql_str = build_sql_string(gold_sql_dict, headers, types)

    # First check: does it execute at all?
    pred_results, pred_error = execute_sql(table, predicted_sql)
    gold_results, gold_error = execute_sql(table, gold_sql_str)

    if pred_error:
        return {
            "category": "syntax_error",
            "details": pred_error,
            "predicted_sql": predicted_sql,
            "gold_sql": gold_sql_str,
        }

    # Check if correct
    from src.eval.execution_accuracy import normalize_result

    if normalize_result(pred_results) == normalize_result(gold_results):
        return {
            "category": "correct",
            "details": None,
            "predicted_sql": predicted_sql,
            "gold_sql": gold_sql_str,
        }

    # --- Incorrect result: classify the error type ---
    errors = []

    # Parse predicted SQL
    pred_agg, pred_col = parse_select(predicted_sql)
    gold_agg_str = AGG_OPS[gold_sql_dict["agg"]]
    gold_col = headers[gold_sql_dict["sel"]]

    # Check SELECT column
    if pred_col.lower() != gold_col.lower():
        errors.append("wrong_column")

    # Check aggregation
    if pred_agg.upper() != gold_agg_str.upper():
        errors.append("wrong_agg")

    # Parse WHERE clauses
    pred_wheres = parse_where(predicted_sql)
    gold_cond_cols = gold_sql_dict["conds"]["column_index"]
    gold_cond_ops = gold_sql_dict["conds"]["operator_index"]
    gold_cond_vals = gold_sql_dict["conds"]["condition"]
    n_gold = len(gold_cond_cols)
    n_pred = len(pred_wheres)

    if n_gold > 0 and n_pred == 0:
        errors.append("missing_where")
    elif n_gold == 0 and n_pred > 0:
        errors.append("extra_where")
    elif n_gold > 0 and n_pred > 0:
        # Compare WHERE details
        gold_where_cols = [headers[i].lower() for i in gold_cond_cols]
        pred_where_cols = [w["col"].lower() for w in pred_wheres]

        gold_where_ops = [["=", ">", "<"][i] for i in gold_cond_ops]
        pred_where_ops = [w["op"] for w in pred_wheres]

        gold_where_vals = [str(v).lower() for v in gold_cond_vals]
        pred_where_vals = [w["val"].lower() for w in pred_wheres]

        if n_pred < n_gold:
            errors.append("missing_where")
        elif n_pred > n_gold:
            errors.append("extra_where")

        if sorted(pred_where_cols) != sorted(gold_where_cols):
            errors.append("wrong_where_col")
        elif sorted(pred_where_ops) != sorted(gold_where_ops):
            errors.append("wrong_where_op")
        elif sorted(pred_where_vals) != sorted(gold_where_vals):
            errors.append("wrong_where_val")

    # Determine final category
    if len(errors) == 0:
        # Parsed components look the same but results differ
        # (formatting or edge-case differences)
        category = "wrong_result_other"
    elif len(errors) == 1:
        category = errors[0]
    else:
        category = "multiple_errors"

    return {
        "category": category,
        "details": errors,
        "predicted_sql": predicted_sql,
        "gold_sql": gold_sql_str,
    }


def analyze_predictions(
    predictions: list[str],
    dataset: list[dict],
) -> dict:
    """
    Run error analysis on a full set of predictions.

    Args:
        predictions: list of predicted SQL strings
        dataset: list of WikiSQL examples (with sql and table fields)

    Returns:
        dict with:
            - summary: {category: count} totals
            - examples: list of per-example classification dicts
            - total: total count
    """
    summary = {}
    examples = []

    for i, (pred, example) in enumerate(zip(predictions, dataset)):
        result = classify_error(pred, example["sql"], example["table"])
        result["index"] = i
        result["question"] = example["question"]
        examples.append(result)

        cat = result["category"]
        summary[cat] = summary.get(cat, 0) + 1

    return {
        "summary": summary,
        "examples": examples,
        "total": len(predictions),
    }


if __name__ == "__main__":
    from datasets import load_dataset

    print("Loading dataset...")
    ds = load_dataset("Salesforce/wikisql")
    test_examples = list(ds["test"].select(range(100)))

    # Sanity check: gold SQL as predictions should all be "correct"
    gold_preds = [
        build_sql_string(ex["sql"], ex["table"]["header"], ex["table"]["types"])
        for ex in test_examples
    ]
    result = analyze_predictions(gold_preds, test_examples)
    print(f"Sanity check (gold SQL): {result['summary']}")
    assert result["summary"].get("correct", 0) == len(test_examples), (
        f"Expected all correct, got: {result['summary']}"
    )
    print("Sanity check passed — all 100 classified as 'correct'.")
