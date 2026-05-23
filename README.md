# Simple DDPM - Diffusion Model from Scratch

A complete implementation of **Denoising Diffusion Probabilistic Model (DDPM)** from scratch, trained on MNIST digits.

---

## 📁 Project Structure
```text
SimpleDDPM/
├── config.py # Hyperparameters
├── ddpm.py # Core diffusion (forward/reverse)
├── train.py # Training script
├── sample.py # Image generation
├── animate_sample.py # Create denoising animation
├── models/
│ └── unet.py # UNet architecture
└── samples/ # Generated images (local only)
```
---

## 🚀 Quick Start
```bash
python train.py

python sample.py

python animate_sample.py
```
