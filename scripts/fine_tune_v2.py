"""
Fine-tuning Script v2  —  Llama-3-8B + QLoRA on WikiSQL
=========================================================
Changes from v1:
  1. FIXED  Chat template mismatch: use formatting_func to bypass TRL's auto
            chat-template wrapping → training format now matches inference format
  2. TUNED  learning_rate: 2e-4 → 1e-4  (less aggressive, avoids memorisation)
  3. TUNED  lora_alpha: 32 → 16  (scaling factor 1.0, more stable)
  4. TUNED  num_train_epochs: 3 → 2  (val loss was rising after epoch 1 in v1)
  5. ADDED  warmup_steps: 200  (stabilises early training)
  6. TUNED  eval_strategy: "epoch" → "steps" every 500  (finer-grained checkpointing)
  7. TUNED  per_device_train_batch_size: 16 → 4  (safe for T4 16 GB)
            gradient_accumulation: 2 → 4  (keeps effective batch size = 16)
  8. SAFE   max_seq_length: 256 → 512  (0% truncation either way, but covers
            the few max-length outliers + chat token overhead)
"""


# ── CELL 1: Install ────────────────────────────────────────────────────────────
# !pip install -q "datasets<3.0" transformers peft trl accelerate bitsandbytes wandb torch


# ── CELL 2: Auth ───────────────────────────────────────────────────────────────
from huggingface_hub import login
import wandb

login()           # HuggingFace token
wandb.login()     # WandB token


# ── CELL 3: Load data ──────────────────────────────────────────────────────────
import json
from google.colab import files

uploaded = files.upload()   # upload train.jsonl and validation.jsonl

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl("train.jsonl")
val_data   = load_jsonl("validation.jsonl")
print(f"Train: {len(train_data):,}  |  Val: {len(val_data):,}")
print(f"\nSample:\n{train_data[0]['text']}")


# ── CELL 4: Load model + tokenizer ────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)
print(f"Loaded. Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


# ── CELL 5: LoRA config ────────────────────────────────────────────────────────
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,          # CHANGED: was 32 → scaling factor now 1.0
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ── CELL 6: Build datasets ─────────────────────────────────────────────────────
from datasets import Dataset

train_dataset = Dataset.from_list([{"text": ex["text"]} for ex in train_data])
val_dataset   = Dataset.from_list([{"text": ex["text"]} for ex in val_data])


# ── CELL 7: Train ──────────────────────────────────────────────────────────────
from trl import SFTConfig, SFTTrainer

# ── KEY FIX: formatting_func bypasses TRL's automatic chat-template wrapping.
#    TRL applies Llama-3's chat template when it sees processing_class=tokenizer
#    + Instruct model. Without this, training format ≠ inference format.
#    With this function, the model is trained on the raw ### Input / ### SQL text,
#    which exactly matches what notebook 03 sends at inference time.
def formatting_func(examples):
    return examples["text"]

training_args = SFTConfig(
    output_dir="./results_v2",

    # Schedule
    num_train_epochs=2,          # CHANGED: was 3 — val loss rose after epoch 1 in v1
    warmup_steps=200,            # ADDED: stabilises early steps (~3% of total)

    # Batch  (T4-safe: 4 × 4 accum = effective batch 16)
    per_device_train_batch_size=4,   # CHANGED: was 16 (A100) → 4 for T4 16 GB
    gradient_accumulation_steps=4,   # CHANGED: was 2 → 4, keeps effective batch = 16

    # Optimiser
    learning_rate=1e-4,          # CHANGED: was 2e-4 — less aggressive
    fp16=True,

    # Evaluation & checkpointing
    logging_steps=50,
    eval_strategy="steps",       # CHANGED: was "epoch" → finer-grained
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Sequence length
    max_seq_length=512,          # SAFE: 0% truncation at 256 too, but 512 covers all

    # Logging
    report_to="wandb",
    run_name="llama3-wikisql-qlora-v2-t4",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    formatting_func=formatting_func,  # KEY FIX: disables auto chat-template
    processing_class=tokenizer,
)

# Sanity check: confirm no chat template was applied to the first example
sample_ids = trainer.train_dataset[0]["input_ids"]
decoded = tokenizer.decode(sample_ids, skip_special_tokens=False)
print("=== Verify training format (should NOT contain <|start_header_id|>) ===")
print(decoded[:300])
assert "<|start_header_id|>" not in decoded, \
    "❌ Chat template still applied! Check TRL version."
print("✅ Format looks correct — raw ### Input / ### SQL format confirmed.\n")

wandb.init(project="text-to-sql-llama", name="llama3-wikisql-qlora-v2")
print("Starting training...")
trainer.train()
print("Training complete!")


# ── CELL 8: Save & push ────────────────────────────────────────────────────────
ADAPTER_REPO = "YOUR_HF_USERNAME/llama3-8b-wikisql-qlora-v2"  # ← change this

model.save_pretrained("./llama3-wikisql-qlora-v2")
tokenizer.save_pretrained("./llama3-wikisql-qlora-v2")
print("Saved locally.")

model.push_to_hub(ADAPTER_REPO)
tokenizer.push_to_hub(ADAPTER_REPO)
print(f"Pushed to Hub: {ADAPTER_REPO}")


# ── CELL 9: Save training results ─────────────────────────────────────────────

# Collect epoch-level results from trainer log history
epoch_results = []
log_history = trainer.state.log_history
eval_logs = [l for l in log_history if "eval_loss" in l]

for log in eval_logs:
    epoch_results.append({
        "step":       log.get("step"),
        "epoch":      log.get("epoch"),
        "val_loss":   log.get("eval_loss"),
    })

# Find the matching train loss (closest step)
train_logs = [l for l in log_history if "loss" in l and "eval_loss" not in l]
for er in epoch_results:
    close = min(train_logs, key=lambda l: abs(l["step"] - er["step"]))
    er["train_loss"] = close.get("loss")

training_results = {
    "model":   "meta-llama/Meta-Llama-3-8B-Instruct",
    "adapter": ADAPTER_REPO,
    "version": "v2",
    "dataset": "wikisql",
    "changes_from_v1": [
        "Fixed chat-template mismatch via formatting_func",
        "learning_rate 2e-4 → 1e-4",
        "lora_alpha 32 → 16",
        "num_epochs 3 → 2",
        "warmup_steps 200 added",
        "eval_strategy epoch → steps/500",
    ],
    "hyperparameters": {
        "lora_r":                     16,
        "lora_alpha":                 16,
        "learning_rate":              1e-4,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "effective_batch_size":       16,
        "max_seq_length":             512,
        "warmup_steps":               200,
        "num_epochs":                 2,
    },
    "results": epoch_results,
}

with open("training_results_v2.json", "w") as f:
    json.dump(training_results, f, indent=2)

print("\nTraining results summary:")
for r in epoch_results:
    print(f"  step {r['step']:>5} | epoch {r['epoch']:.1f} "
          f"| train_loss {r['train_loss']:.4f} | val_loss {r['val_loss']:.4f}")
