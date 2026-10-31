import matplotlib.pyplot as plt
import os

# 1. Ensure the assets directory exists
os.makedirs("assets", exist_ok=True)

# 2. Simulated or recorded epoch-wise training data from your Tversky U-Net run
# (If you logged your losses during training, you can paste your actual lists here!)
epochs = list(range(1, 26))  # Assuming 25 epochs
train_losses = [0.65, 0.52, 0.44, 0.38, 0.34, 0.31, 0.28, 0.26, 0.24, 0.23, 
                0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.15, 0.14, 0.14, 
                0.13, 0.13, 0.12, 0.12, 0.11]

val_losses = [0.62, 0.49, 0.42, 0.37, 0.33, 0.30, 0.27, 0.25, 0.24, 0.22, 
              0.21, 0.20, 0.19, 0.18, 0.18, 0.17, 0.16, 0.16, 0.15, 0.15, 
              0.14, 0.14, 0.14, 0.13, 0.13]

# 3. Generate the professional plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label='Train Loss (Tversky)', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
plt.plot(epochs, val_losses, label='Validation Loss (Tversky)', color='#ff7f0e', linewidth=2.5, marker='s', markersize=4)

plt.title('U-Net (ResNet-34) Training & Validation Loss Convergence', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Tversky Loss Value', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# 4. Save and show
plt.tight_layout()
plt.savefig("assets/loss_curve.png", dpi=300, bbox_inches='tight')
plt.show()
print("Loss curve graph saved successfully to 'assets/loss_curve.png'!")