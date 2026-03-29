
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

from config import *
from ddpm import DDPM
from models.unet import UNet


def train():
    
    print("=" * 50)
    print("STARTING DDPM TRAINING")
    print("=" * 50)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("\n[1/5] Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    print(f"   Dataset size: {len(train_dataset)} images")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Number of batches: {len(train_loader)}")
    
    print("\n[2/5] Initializing model...")
    model = UNet(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        base_channels=64,
        time_emb_dim=256
    ).to(DEVICE)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n[3/5] Initializing DDPM and optimizer...")
    ddpm = DDPM()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n[4/5] Training for {NUM_EPOCHS} epochs...")
    print("-" * 50)
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(DEVICE)
            batch_size = images.shape[0]
            
            t = torch.randint(0, NUM_TIMESTEPS, (batch_size,), device=DEVICE)
            
            noisy_images, true_noise = ddpm.add_noise(images, t)
            
            predicted_noise = model(noisy_images, t)
            
            loss = nn.MSELoss()(predicted_noise, true_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Average Loss: {avg_loss:.6f}")
        
        checkpoint_path = f"{CHECKPOINT_DIR}/ddpm_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"   Saved checkpoint: {checkpoint_path}")
    
    print("\n[5/5] Training completed!")
    print("=" * 50)
    
    final_model_path = f"{MODELS_DIR}/ddpm_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    train()