import matplotlib.pyplot as plt

# Loss values from training logs (every 10 steps)
steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
loss =  [0.7172, 0.5952, 0.6766, 0.6318, 0.7562, 0.6654, 0.4709, 0.5921, 0.5295, 0.5609]
accuracy = [0.8153, 0.8392, 0.8368, 0.842, 0.8224, 0.8189, 0.8511, 0.8325, 0.8537, 0.8377]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss plot
ax1.plot(steps, loss, marker='o', color='royalblue', linewidth=2, markersize=5)
ax1.set_title('Training Loss', fontsize=14)
ax1.set_xlabel('Steps')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.3, 0.9)

# Accuracy plot
ax2.plot(steps, accuracy, marker='o', color='seagreen', linewidth=2, markersize=5)
ax2.set_title('Mean Token Accuracy', fontsize=14)
ax2.set_xlabel('Steps')
ax2.set_ylabel('Accuracy')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.75, 0.90)

plt.suptitle('LLaMA 3.2-3B QLoRA Fine-tuning — Training Metrics', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
print("Plot saved to training_metrics.png")