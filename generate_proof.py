import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import os
from skimage.io import imread

# 1. Setup device and load your winning Tversky model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights=None,     
    in_channels=3,                  
    classes=1                       
).to(device)

model.load_state_dict(torch.load("models/best_unet_tversky.pth", map_location=device))
model.eval()

os.makedirs("assets", exist_ok=True)

# 2. Load a REAL test image from your dataset folder (update the path to your actual image)
image_path = "TCGA_DU_A5TS_19970726_12.tif" # e.g., your sample .tif file path
mask_path = "TCGA_DU_A5TS_19970726_12_mask.tif"
# If you don't have it handy, you can use any test slice from your dataset

# Read and preprocess the real image
raw_img = imread(image_path)
ground_truth_mask = imread(mask_path)
if raw_img.max() > 1.0:
    img = raw_img.astype(np.float32) / 255.0
else:
    img = raw_img.astype(np.float32)

if len(img.shape) == 2:
    img = np.stack([img, img, img], axis=-1)
elif img.shape[2] > 3:
    img = img[..., :3]

# Run inference
img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.sigmoid(logits)
    pred_mask = (probs > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()

# 3. Generate the clean 3-Panel Comparison Figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Real FLAIR MRI
axes[0].imshow(img[..., 1], cmap='gray')
axes[0].set_title("1. Raw MRI Scan (FLAIR)")
axes[0].axis('off')

# Panel 2: Ground Truth Mask (if you have the corresponding _mask.tif file)
axes[1].imshow(ground_truth_mask, cmap='gray')
axes[1].set_title("2. Ground Truth (Doctor Mask)")
axes[1].axis('off')

# Panel 3: Model Prediction Overlay
axes[2].imshow(img[..., 1], cmap='gray')
axes[2].imshow(pred_mask, cmap='Reds', alpha=0.5)
axes[2].set_title("3. Your Model Prediction (Tversky U-Net)")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("assets/visual_proof_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
print("Real visual proof plot saved to 'assets/visual_proof_comparison.png'!")