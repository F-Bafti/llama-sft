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

print("\nBaseline outputs saved to baseline_output.txt")