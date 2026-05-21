import torch
import json
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Model and dataset
MODEL_ID = "meta-llama/Llama-3.2-3B"
DATASET_ID = "iamtarun/python_code_instructions_18k_alpaca"

# LoRA config
LORA_R = 8           # Rank — how many dimensions the adapter matrices have
LORA_ALPHA = 16      # Scaling factor — usually 2x rank
LORA_DROPOUT = 0.05  # Regularization — prevents overfitting

# Training config
MAX_SEQ_LENGTH = 512   # Max tokens per sample
BATCH_SIZE = 2         # Small batch to fit in 12GB VRAM
GRAD_ACCUM = 4         # Simulates batch size of 8 (2x4)
LEARNING_RATE = 2e-4   # Standard for LoRA fine-tuning
NUM_STEPS = 1000       # Small run for demo purposes
OUTPUT_DIR = "./llama-sft-output"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",      # NormalFloat4 — best for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16 for stability
    bnb_4bit_use_double_quant=True  # Quantize the quantization constants too — saves extra memory
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention layers
        "gate_proj", "up_proj", "down_proj"        # MLP layers
    ]
)

# Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
dataset = load_dataset(DATASET_ID, split="train")

# Format each sample into instruction format
def format_prompt(sample):
    return {
        "prompt": f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n",
        "completion": sample['output']
    }

dataset = dataset.map(format_prompt)
print(f"Dataset size: {len(dataset)} samples")

# Training config
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=NUM_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    max_length=MAX_SEQ_LENGTH,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)

# Print some examples
print("\n--- Dataset Examples ---")
for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"PROMPT:\n{dataset[i]['prompt']}")
    print(f"COMPLETION:\n{dataset[i]['completion']}")
    print("-" * 50)


# Track metrics
metrics_log = []

class MetricsCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            metrics_log.append({
                "step": state.global_step,
                "loss": logs.get("loss"),
                "grad_norm": logs.get("grad_norm"),
                "learning_rate": logs.get("learning_rate"),
                "entropy": logs.get("entropy"),
                "mean_token_accuracy": logs.get("mean_token_accuracy"),
            })
            # Save to file after every log
            with open("training_metrics.json", "w") as f:
                json.dump(metrics_log, f, indent=2)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    callbacks=[MetricsCallback()]  # ← add this
)

# Start training
print("Starting training...")
trainer.train()

# Save the LoRA adapters
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")