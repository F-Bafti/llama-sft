import json
import matplotlib.pyplot as plt

# Load metrics from file
# Load metrics from file
with open("training_metrics.json", "r") as f:
    metrics = json.load(f)

# Filter out entries where loss or accuracy is None
filtered = [m for m in metrics if m["loss"] is not None and m["mean_token_accuracy"] is not None]

steps =    [m["step"] for m in filtered]
loss =     [m["loss"] for m in filtered]
accuracy = [m["mean_token_accuracy"] for m in filtered]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss plot
ax1.plot(steps, loss, marker='o', color='royalblue', linewidth=2, markersize=5)
ax1.set_title('Training Loss', fontsize=14)
ax1.set_xlabel('Steps')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.0, 1.0)


# Accuracy plot
ax2.plot(steps, accuracy, marker='o', color='seagreen', linewidth=2, markersize=5)
ax2.set_title('Mean Token Accuracy', fontsize=14)
ax2.set_xlabel('Steps')
ax2.set_ylabel('Accuracy')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.0, 1.0)

plt.suptitle('LLaMA 3.2-3B QLoRA Fine-tuning — Training Metrics', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
print("Plot saved to training_metrics.png")