
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torchvision.utils import save_image
import os
import numpy as np

from config import *
from ddpm import DDPM
from models.unet import UNet


def create_animation(model_path=None, num_frames=20, save_dir="animation"):
    print("=" * 50)
    print("CREATING DENOISING ANIMATION")
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
        print(f"   ⚠️ Model not found!")
        return
    
    ddpm = DDPM()
    model.eval()
    
    print("\n[2/4] Creating random noise...")
    x = torch.randn(1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    
    print(f"\n[3/4] Denoising and capturing {num_frames} frames...")
    frames = []
    timesteps_to_save = []
    
    for i in range(num_frames):
        t_step = int(999 * (1 - i / (num_frames - 1)))
        timesteps_to_save.append(t_step)
    
    print(f"   Will capture at timesteps: {timesteps_to_save[:5]}...{timesteps_to_save[-5:]}")
    
    with torch.no_grad():
        for t in reversed(range(NUM_TIMESTEPS)):
            t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)
            predicted_noise = model(x, t_tensor)
            x = ddpm.sample_step(model, x, t_tensor, predicted_noise)
            
            # Lưu frame nếu ở timestep cần
            if t in timesteps_to_save:
                frame = x.clone().cpu()
                frames.append(frame)
                print(f"   Captured at t={t} ({len(frames)}/{num_frames})")
    
    frames = list(reversed(frames))

    print("\n[4/4] Creating animation...")
    frames_normalized = []
    for frame in frames:
        frame_norm = (frame + 1) / 2
        frame_norm = torch.clamp(frame_norm, 0, 1)
        frames_normalized.append(frame_norm.squeeze().numpy())
    
    for i, frame in enumerate(frames_normalized):
        plt.imsave(f"{save_dir}/frame_{i:03d}.png", frame, cmap='gray')
    print(f"   Saved {len(frames_normalized)} frames to {save_dir}/")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    img_display = ax.imshow(frames_normalized[0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"Step: 0/{NUM_TIMESTEPS-1}")
    ax.axis('off')
    
    def update(frame_idx):
        img_display.set_array(frames_normalized[frame_idx])
        step = timesteps_to_save[frame_idx]
        ax.set_title(f"Step: {step}/{NUM_TIMESTEPS-1}")
        return [img_display]
    
    anim = FuncAnimation(
        fig, update, frames=len(frames_normalized),
        interval=200, blit=True, repeat=True
    )
    
    anim_path = f"{save_dir}/denoising_animation.gif"
    anim.save(anim_path, writer='pillow', fps=5)
    print(f"   Saved animation: {anim_path}")
    
    plt.show()
    
    print("\n" + "=" * 50)
    print(f"✅ ANIMATION COMPLETE!")
    print(f"   Frames saved to: {save_dir}/")
    print(f"   Animation saved to: {anim_path}")
    print("=" * 50)
    
    return frames_normalized, timesteps_to_save


def create_grid_animation(model_path=None, num_samples=4, save_dir="animation_grid"):
    """
    Tạo animation cho nhiều ảnh cùng lúc
    """
    print("=" * 50)
    print("CREATING GRID ANIMATION")
    print("=" * 50)
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = UNet(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        base_channels=64,
        time_emb_dim=256
    ).to(DEVICE)
    
    if model_path is None:
        model_path = f"{MODELS_DIR}/ddpm_final.pth"
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"   ✅ Loaded model from: {model_path}")
    
    ddpm = DDPM()
    model.eval()
    
    print(f"\n[2/3] Creating {num_samples} random noise images...")
    x = torch.randn(num_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    
    timesteps_to_save = [999, 800, 600, 400, 200, 100, 50, 20, 5, 0]
    frames = {t: [] for t in timesteps_to_save}
    
    print(f"\n[3/3] Denoising and capturing frames...")
    with torch.no_grad():
        for t in reversed(range(NUM_TIMESTEPS)):
            t_tensor = torch.full((num_samples,), t, device=DEVICE, dtype=torch.long)
            predicted_noise = model(x, t_tensor)
            x = ddpm.sample_step(model, x, t_tensor, predicted_noise)
            
            if t in timesteps_to_save:
                frame = x.clone().cpu()
                frames[t].append(frame)
                print(f"   Captured at t={t}")
    
    print("\nCreating grid images...")
    for t in timesteps_to_save:
        if frames[t]:
            grid = frames[t][0]
            grid = (grid + 1) / 2
            grid = torch.clamp(grid, 0, 1)
            
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                if i < num_samples:
                    img = grid[i].squeeze().numpy()
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
            plt.suptitle(f"Denoising Step: {t}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/grid_t{t}.png")
            plt.close()
    
    print(f"   Saved {len(timesteps_to_save)} grid images to {save_dir}/")
    
    print("\n" + "=" * 50)
    print("✅ GRID ANIMATION COMPLETE!")
    print(f"   Run: python -c \"import imageio; images = [imageio.imread(f'{save_dir}/grid_t{t}.png') for t in {timesteps_to_save}]; imageio.mimsave('{save_dir}/grid_animation.gif', images, fps=2)\"")
    print("=" * 50)


if __name__ == "__main__":
    create_animation(num_frames=30)
    