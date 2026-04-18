"""
Evaluation Script v2  —  evaluates the new QLoRA v2 adapter
Only changes from original notebook 03:
  - adapter_id updated to v2 HF repo
  - results saved as ft_v2_results.json
  - comparison table updated with v2 label
Everything else (generate_sql, execute_sql, build_db, etc.) is unchanged.
"""


# ── CELL 1: Install ────────────────────────────────────────────────────────────
# !pip install -q "datasets<3.0" transformers peft bitsandbytes accelerate torch


# ── CELL 2: Auth ───────────────────────────────────────────────────────────────
from huggingface_hub import login
login()


# ── CELL 3: Load model  ── ONLY CHANGE: adapter_id points to v2 ───────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id   = "meta-llama/Meta-Llama-3-8B-Instruct"
adapter_id = "YOUR_HF_USERNAME/llama3-8b-wikisql-qlora-v2"  # ← change to your v2 repo

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()
print(f"Done! Memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


# ── CELL 4: Load dataset + baseline results ────────────────────────────────────
import json
from datasets import load_dataset
from google.colab import files

ds = load_dataset("Salesforce/wikisql", trust_remote_code=True)
examples = list(ds['test'].select(range(500)))

uploaded = files.upload()  # upload base_model_results.json

with open("base_model_results.json") as f:
    base_results = json.load(f)

base_predictions = base_results["predictions"]
print(f"Loaded {len(base_predictions)} base model predictions")
print(f"Base model execution accuracy: {base_results['execution_accuracy']:.1%}")


# ── CELL 5: generate_sql (unchanged from original) ────────────────────────────
import re

def fix_column_names(sql: str, columns: list) -> str:
    for col in sorted(columns, key=len, reverse=True):
        if not col:
            continue
        quoted = f'`{col}`'
        underscore_variant = re.sub(r'[\s\-–]+', '_', col)
        variants = [col]
        if underscore_variant.lower() != col.lower():
            variants.append(underscore_variant)
        for variant in variants:
            if re.search(re.escape(quoted), sql, re.IGNORECASE):
                break
            if re.search(re.escape(variant), sql, re.IGNORECASE):
                sql = re.sub(re.escape(variant), quoted, sql, flags=re.IGNORECASE)
                break
    return sql


def generate_sql(question: str, columns: list, types: list, max_new_tokens: int = 128) -> str:
    col_defs = ", ".join(f"{name} ({dtype})" for name, dtype in zip(columns, types))
    prompt = (
        f"### Input:\n"
        f"Columns: {col_defs}\n\n"
        f"Question: {question}\n\n"
        f"### SQL:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if "```" in response:
        lines = response.split('\n')
        response = ' '.join(l for l in lines if not l.strip().startswith('```')).strip()

    for line in response.split('\n'):
        line = line.strip()
        if line.upper().startswith('SELECT'):
            line = line.split(';')[0].strip()
            if '--' in line:
                line = line[:line.index('--')].strip()
            if '###' in line:
                line = line[:line.index('###')].strip()
            line = re.sub(r'\bFROM\s+\w+\b', 'FROM table', line, flags=re.IGNORECASE)
            line = fix_column_names(line, columns)
            line = re.sub(r"\s+AND\s+`[^`]+`\s*=\s*''", '', line, flags=re.IGNORECASE)
            return line

    return response.split('\n')[0].split(';')[0].strip()


# Quick sanity check on one example
example = ds['test'][0]
predicted = generate_sql(example['question'], example['table']['header'], example['table']['types'])
print(f"Question:  {example['question']}")
print(f"Predicted: {predicted}")
print(f"Gold:      {example['sql']['human_readable']}")


# ── CELL 6: Generate predictions for all 500 examples ─────────────────────────
from tqdm.notebook import tqdm

ft_predictions = []
for example in tqdm(examples):
    pred = generate_sql(
        example['question'],
        example['table']['header'],
        example['table']['types']
    )
    ft_predictions.append(pred)

print(f"Generated {len(ft_predictions)} predictions")
print("\nSample predictions:")
for i in range(3):
    print(f"  {i+1}. {ft_predictions[i]}")


# ── CELL 7: Evaluate (unchanged from original) ────────────────────────────────
import sqlite3

def build_db(table):
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    headers = table["header"]
    types = table["types"]
    type_map = {"text": "TEXT", "number": "REAL", "real": "REAL"}
    col_defs = ", ".join(f'`{col}` {type_map.get(t, "TEXT")}' for col, t in zip(headers, types))
    cursor.execute(f"CREATE TABLE data ({col_defs})")
    for row in table["rows"]:
        converted = []
        for val, col_type in zip(row, types):
            if col_type in ("real", "number"):
                try:
                    converted.append(float(str(val).replace(",", "")))
                except:
                    converted.append(val)
            else:
                converted.append(val)
        placeholders = ", ".join(["?"] * len(converted))
        cursor.execute(f"INSERT INTO data VALUES ({placeholders})", converted)
    conn.commit()
    return conn

def execute_sql(table, sql):
    sql = sql.replace("FROM table", "FROM data")
    try:
        conn = build_db(table)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return [], str(e)

def normalize_result(result):
    return sorted([str(row) for row in result])

def build_sql_string(sql, columns, types):
    AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    COND_OPS = ["=", ">", "<"]
    agg = AGG_OPS[sql["agg"]]
    sel_col = columns[sql["sel"]]
    select_clause = f"SELECT {agg}(`{sel_col}`)" if agg else f"SELECT `{sel_col}`"
    where_clauses = []
    for col_idx, op_idx, cond in zip(
        sql["conds"]["column_index"],
        sql["conds"]["operator_index"],
        sql["conds"]["condition"],
    ):
        col = columns[col_idx]
        op = COND_OPS[op_idx]
        col_type = types[col_idx]
        if col_type in ("real", "number"):
            cleaned = str(cond).replace(",", "")
            try:
                float(cleaned)
                where_clauses.append(f"`{col}` {op} {cleaned}")
            except:
                where_clauses.append(f"`{col}` {op} '{str(cond).replace(chr(39), chr(39)*2)}'")
        else:
            escaped = str(cond).replace("'", "''")
            where_clauses.append(f"`{col}` {op} '{escaped}'")
    where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    return select_clause + " FROM table" + where_clause

ft_total        = len(ft_predictions)
ft_correct      = 0
ft_syntax_errors = 0

for pred, example in zip(ft_predictions, examples):
    table    = example["table"]
    gold_sql = build_sql_string(example["sql"], table["header"], table["types"])
    pred_results, pred_error = execute_sql(table, pred)
    gold_results, _          = execute_sql(table, gold_sql)
    if pred_error:
        ft_syntax_errors += 1
    elif normalize_result(pred_results) == normalize_result(gold_results):
        ft_correct += 1

# ── Final comparison table ─────────────────────────────────────────────────────
print("=" * 55)
print(f"{'Method':<30} {'Exec Acc':>10} {'Syntax Err':>12}")
print("=" * 55)
print(f"{'Base Llama-3-8B':<30} {base_results['execution_accuracy']:>10.1%} {base_results['syntax_error_rate']:>12.1%}")
print(f"{'+ QLoRA Fine-tune v1':<30} {'25.4%':>10} {'27.4%':>12}")
print(f"{'+ QLoRA Fine-tune v2':<30} {ft_correct/ft_total:>10.1%} {ft_syntax_errors/ft_total:>12.1%}")
print("=" * 55)
improvement = ft_correct / ft_total - base_results['execution_accuracy']
print(f"\nv2 vs Baseline improvement: {improvement:+.1%}")
print(f"v2 vs v1 improvement:       {ft_correct/ft_total - 0.254:+.1%}")


# ── CELL 8: Save results + download ───────────────────────────────────────────
ft_v2_results = {
    "model":          "meta-llama/Meta-Llama-3-8B-Instruct",
    "adapter":        adapter_id,
    "version":        "v2",
    "dataset":        "wikisql",
    "split":          "test",
    "n_examples":     ft_total,
    "execution_accuracy":  ft_correct / ft_total,
    "syntax_error_rate":   ft_syntax_errors / ft_total,
    "correct":        ft_correct,
    "syntax_errors":  ft_syntax_errors,
    "predictions":    ft_predictions,
}

with open("ft_v2_results.json", "w") as f:
    json.dump(ft_v2_results, f, indent=2)

print("Saved ft_v2_results.json")

from google.colab import files
files.download("ft_v2_results.json")
