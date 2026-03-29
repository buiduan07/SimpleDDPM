"""
Generate ảnh từ noise bằng diffusion model đã train
"""

import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

from config import *
from ddpm import DDPM
from models.unet import UNet


def generate_samples(model_path=None, num_samples=16, save_dir="samples"):
    print("=" * 50)
    print("GENERATING SAMPLES FROM DDPM")
    print("=" * 50)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n[1/4] Loading model...")
    model = UNet(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        base_channels=64,
        time_emb_dim=256
    ).to(DEVICE)
    
    if model_path is None:
        model_path = f"{MODELS_DIR}/ddpm_final.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"   ✅ Loaded model from: {model_path}")
    else:
        print(f"   ⚠️ Model not found: {model_path}")
        return
    
    ddpm = DDPM()
    model.eval()
    
    print(f"\n[2/4] Creating {num_samples} random noise images...")
    x = torch.randn(num_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    
    print(f"\n[3/4] Denoising from t={NUM_TIMESTEPS-1} to 0...")
    print("   This will take ~30-60 seconds...")
    
    with torch.no_grad():
        for t in reversed(range(NUM_TIMESTEPS)):
            t_tensor = torch.full((num_samples,), t, device=DEVICE, dtype=torch.long)
            predicted_noise = model(x, t_tensor)
            x = ddpm.sample_step(model, x, t_tensor, predicted_noise)
            
            if t % 100 == 0:
                print(f"   Step {t}/{NUM_TIMESTEPS-1}")
    
    print("\n[4/4] Saving images...")
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    
    for i in range(num_samples):
        save_path = f"{save_dir}/sample_{i}.png"
        save_image(x[i], save_path)
        print(f"   Saved: {save_path}")
    
    print("\nDisplaying samples...")
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = x[i].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
    plt.suptitle(f"Generated MNIST digits (after {NUM_EPOCHS} epochs)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/samples_grid.png")
    plt.show()
    
    print("\n" + "=" * 50)
    print("✅ DONE! Check the 'samples' folder for images.")
    print("=" * 50)


if __name__ == "__main__":
    generate_samples()