import json
from datasets import load_dataset

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<"]


def build_sql_string(sql: dict, columns: list[str]) -> str:
    """Convert WikiSQL structured SQL dict to a human-readable SQL string."""
    agg = AGG_OPS[sql["agg"]]
    sel_col = columns[sql["sel"]]

    if agg:
        select_clause = f"SELECT {agg}({sel_col})"
    else:
        select_clause = f"SELECT {sel_col}"

    where_clauses = []
    for col_idx, op_idx, cond in zip(
        sql["conds"]["column_index"],
        sql["conds"]["operator_index"],
        sql["conds"]["condition"],
    ):
        col = columns[col_idx]
        op = COND_OPS[op_idx]
        where_clauses.append(f"`{col}` {op} '{cond}'")

    if where_clauses:
        where_clause = " WHERE " + " AND ".join(where_clauses)
    else:
        where_clause = ""

    return select_clause + " FROM table" + where_clause


def format_prompt(question: str, column_names: list[str], column_types: list[str]) -> str:
    """Format the input prompt for the model."""
    col_defs = ", ".join(
        f"{name} ({dtype})" for name, dtype in zip(column_names, column_types)
    )
    return (
        f"### Input:\n"
        f"Columns: {col_defs}\n\n"
        f"Question: {question}\n\n"
        f"### SQL:\n"
    )


def convert_example(example: dict) -> dict:
    """Convert a single WikiSQL example to a prompt-completion pair."""
    question = example["question"]
    columns = example["table"]["header"]
    types = example["table"]["types"]
    sql = example["sql"]

    prompt = format_prompt(question, columns, types)
    completion = build_sql_string(sql, columns)

    return {"prompt": prompt, "completion": completion, "text": prompt + completion}


def prepare_and_save(output_dir: str = "data/processed") -> None:
    """Load WikiSQL, convert all splits, and save as JSONL files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("Salesforce/wikisql", trust_remote_code=True)

    for split in ["train", "validation", "test"]:
        output_path = f"{output_dir}/{split}.jsonl"
        count = 0
        with open(output_path, "w") as f:
            for example in ds[split]:
                converted = convert_example(example)
                f.write(json.dumps(converted) + "\n")
                count += 1
        print(f"Saved {count} examples to {output_path}")


if __name__ == "__main__":
    prepare_and_save()