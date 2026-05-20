import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-3B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # half precision to save VRAM
    device_map="auto"           # automatically puts model on GPU
)

# Test prompt
prompt = "Write a Python function that reverses a string."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False,        # greedy decoding — more deterministic
    repetition_penalty=1.3  # penalizes repeating the same tokens
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Save baseline output
with open("baseline_output.txt", "w") as f:
    f.write(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\nBaseline output saved to baseline_output.txt")