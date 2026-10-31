import time
import torch
import segmentation_models_pytorch as smp

# 1. Setup device and load your winning Tversky model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running benchmark on device: {device}")

model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights=None,     
    in_channels=3,                  
    classes=1                       
).to(device)

# Load your local weights file
model.load_state_dict(torch.load("models/best_unet_tversky.pth", map_location=device))
model.eval()
print("Model successfully loaded!")

# 2. Create a dummy input tensor matching your slice dimensions (1, 3, 256, 256)
dummy_input = torch.randn(1, 3, 256, 256).to(device)

# 3. Warm-up passes to stabilize hardware timings
print("Warming up model...")
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input)

# 4. Benchmark execution time over 100 iterations
print("Running latency benchmark...")
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)
end_time = time.time()

# 5. Calculate and print average latency
avg_latency = (end_time - start_time) / 100 * 1000  # Converted to milliseconds

print("\n" + "="*40)
print(f"Average Inference Latency: {avg_latency:.2f} ms per slice")
print("="*40)