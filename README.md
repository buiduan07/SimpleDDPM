# Simple DDPM - Diffusion Model from Scratch

A complete implementation of **Denoising Diffusion Probabilistic Model (DDPM)** from scratch, trained on MNIST digits.

## 🎯 What I Learned
- Forward diffusion process: adding Gaussian noise to images
- Reverse diffusion: denoising to generate new images
- UNet architecture with skip connections and time embeddings
- Training diffusion models from scratch
- Sampling and visualization

## 📊 Results
- **Model parameters**: 8.5 million
- **Training**: 5 epochs on MNIST (60,000 images)
- **Final loss**: 0.0256

## 📁 Project Structure
SimpleDDPM/
├── config.py # Hyperparameters
├── ddpm.py # Core diffusion (forward/reverse)
├── train.py # Training script
├── sample.py # Image generation
├── animate_sample.py # Create denoising animation
├── models/
│ └── unet.py # UNet architecture
└── samples/ # Generated images (local only)


## 🚀 Quick Start
```bash
# Train
python train.py

# Generate samples
python sample.py

# Create animation
python animate_sample.py