import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "meta-llama/Llama-3.2-3B"
ADAPTER_DIR = "./llama-sft-output"

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load base model
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load LoRA adapters on top
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

prompts = [
    "Write a Python function that reverses a string.",
    "Write a Python function that checks if a number is prime.",
    "Write a Python function that finds the maximum element in a list."
]

results = []
for i, prompt in enumerate(prompts):
    print(f"\n--- Prompt {i+1}: {prompt} ---")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.3
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    results.append(response)

# Save all baseline outputs
with open("baseline_output.txt", "w") as f:
    for i, result in enumerate(results):
        f.write(f"--- Prompt {i+1} ---\n")
        f.write(result)
        f.write("\n\n")

print("\nOutput saved to finetuned_output.txt")