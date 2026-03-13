import re


def normalize_sql(sql: str) -> str:
    """Normalize a SQL string for comparison."""
    sql = sql.lower().strip()
    sql = re.sub(r"\s+", " ", sql)          # collapse whitespace
    sql = sql.replace("'", '"')              # normalize quotes
    sql = sql.replace("`", "")              # remove backticks
    return sql


def exact_match_single(predicted_sql: str, gold_sql: str) -> bool:
    """Check if two SQL strings are identical after normalization."""
    return normalize_sql(predicted_sql) == normalize_sql(gold_sql)


def exact_match_dataset(predictions: list[str], gold_sqls: list[str]) -> dict:
    """Compute exact match accuracy over a dataset."""
    total = len(predictions)
    matches = sum(
        exact_match_single(pred, gold)
        for pred, gold in zip(predictions, gold_sqls)
    )
    return {
        "exact_match_accuracy": matches / total,
        "total": total,
        "matches": matches,
    }