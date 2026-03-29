
import torch
import torch.nn.functional as F
from config import NUM_TIMESTEPS, BETA_START, BETA_END, DEVICE

class DDPM:
    def __init__(self):
        self.beta = torch.linspace(BETA_START, BETA_END, NUM_TIMESTEPS).to(DEVICE)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        
        print(f"DDPM initialized with {NUM_TIMESTEPS} timesteps on {DEVICE}")

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt, noise

    def sample_step(self, model, xt, t, predicted_noise=None):
        if predicted_noise is None:
            predicted_noise = model(xt, t)
        
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        
        mean = (1 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
        
        if t[0] > 0:
            noise = torch.randn_like(xt)
            sigma_t = torch.sqrt(beta_t)
            return mean + sigma_t * noise
        else:
            return mean

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]