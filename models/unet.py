
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.silu = nn.SiLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch)
            )
        
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.silu(x)

        if t_emb is not None and self.time_mlp is not None:
            t_out = self.time_mlp(t_emb)
            x = x + t_out[:, :, None, None]

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.silu(x)

        return x + residual


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, time_emb_dim=256):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.down1 = Block(in_channels, base_channels, time_emb_dim)
        self.down2 = Block(base_channels, base_channels * 2, time_emb_dim)
        self.down3 = Block(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = Block(base_channels * 4, base_channels * 8, time_emb_dim)

        # Decoder - ConvTranspose2d
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        
        # Blocks sau upsample
        self.block3 = Block(base_channels * 4 + base_channels * 4, base_channels * 4, time_emb_dim)
        self.block2 = Block(base_channels * 2 + base_channels * 2, base_channels * 2, time_emb_dim)
        self.block1 = Block(base_channels + base_channels, base_channels, time_emb_dim)

        # Output
        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)

        # Bottleneck
        b = self.bottleneck(self.pool(d3), t_emb)

        # Decoder với skip connections
        u3 = self.up3(b)
        u3 = F.pad(u3, (0, 1, 0, 1))
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.block3(u3, t_emb)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.block2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.block1(u1, t_emb)

        return self.out(u1)


if __name__ == "__main__":
    device = "cpu"
    model = UNet().to(device)
    x = torch.randn(4, 1, 28, 28).to(device)
    t = torch.randint(0, 1000, (4,)).to(device)
    out = model(x, t)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")