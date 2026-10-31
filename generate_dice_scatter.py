import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("assets", exist_ok=True)

# Simulated Dice scores across 22 test patients (or your actual test results)
np.random.seed(42)
patients = [f"Patient {i+1}" for i in range(22)]
dice_scores = np.random.uniform(0.78, 0.92, size=22)
dice_scores.sort()  # Sort for clean visualization

median_dice = np.median(dice_scores)

# Generate the dot plot
plt.figure(figsize=(10, 5))
plt.scatter(patients, dice_scores, color='#1f77b4', s=70, zorder=3, label='Patient Dice Score')
plt.axhline(y=median_dice, color='red', linestyle='--', linewidth=1.5, label=f'Median Dice ({median_dice:.2f})')

plt.title('Per-Patient Dice Similarity Coefficient (Tversky U-Net Test Set)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Test Patients', fontsize=11)
plt.ylabel('Dice Score', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.ylim(0.70, 1.0)
plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig("assets/dice_scatter_plot.png", dpi=300, bbox_inches='tight')
plt.show()
print("Dice scatter plot saved successfully to 'assets/dice_scatter_plot.png'!")