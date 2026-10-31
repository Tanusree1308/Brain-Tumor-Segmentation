import gradio as gr
import torch
import numpy as np
import segmentation_models_pytorch as smp

# 1. Setup device and load your saved Tversky model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-create the exact ResNet-34 U-Net architecture structure and load saved weights
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load("models/best_unet_tversky.pth", map_location=device))
model.to(device)
model.eval()

def predict_tumor(input_image):
    """
    Takes an input image from Gradio, preprocesses it, runs inference through 
    our trained U-Net, and returns a red overlay image plus the tumor pixel count.
    """
    if input_image is None:
        return None, "No image provided."
        
    # Normalize image to 0.0 - 1.0
    if input_image.max() > 1.0:
        img = input_image.astype(np.float32) / 255.0
    else:
        img = input_image.astype(np.float32)
        
    # Handle shape safety (Ensuring we have 3 channels)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] > 3:
        img = img[..., :3]

    # Prepare tensor for PyTorch: (H, W, C) -> (C, H, W) -> add Batch dim (1, C, H, W)
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        # Threshold at 0.5 to create binary mask
        pred_mask = (probs > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
        
    # Calculate tumor size in pixels
    tumor_pixels = int(pred_mask.sum())
    
    # Use the FLAIR channel (index 1) as our visual grayscale background base
    flair = img[..., 1]
    
    # Convert grayscale FLAIR to a 3-channel RGB image for color painting
    overlay = np.stack([flair, flair, flair], axis=-1)
    
    # Paint the predicted tumor region bright RED [1.0, 0.0, 0.0]
    overlay[pred_mask == 1] = [1.0, 0.0, 0.0]
    
    # Scale back to 0-255 uint8 for Gradio image display
    output_image = (overlay * 255).astype(np.uint8)
    
    status_text = f"Analysis Complete. Detected Tumor Area: {tumor_pixels} pixels."
    return output_image, status_text

# Build the Gradio Web Interface
demo = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="numpy", label="Upload Brain MRI Slice (.tif)"),
    outputs=[
        gr.Image(type="numpy", label="Tumor Segmentation Overlay (Red = Predicted Tumor)"),
        gr.Textbox(label="Diagnostic Summary")
    ],
    title="🧠 Brain Tumor Segmentation (U-Net & PyTorch)",
    description=(
        "Upload a lower-grade glioma MRI slice. The U-Net model will analyze the 3-channel scan, "
        "detect the tumor boundaries, and outline them in red."
    )
)

# Launch the app with a public shareable link
if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(share=True, debug=True)