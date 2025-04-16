import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

@dataclass
class UNetConfig:
    bottleneck_blocks: int = 2
    res_blocks: int = 2  
    depth: int = 3         
    image_size: int = 64
    in_channels: int = 3  
    out_channels: int = 3 
    channels: Tuple = (16, 32, 64, 128)
    embed_dim: int = 64

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(16, out_channels)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(16, out_channels)
        self.emb_proj = nn.Linear(embed_dim, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, emb):
        residual = self.shortcut(x)
        x = self.norm1(self.conv1(x))
        x = self.activation(x)
        emb_out = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        x += emb_out
        x = self.norm2(self.conv2(x))
        x = self.activation(x)
        return x + residual
    
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(16, channels)
        self.q = nn.Conv1d(channels, channels, kernel_size=1)
        self.k = nn.Conv1d(channels, channels, kernel_size=1)
        self.v = nn.Conv1d(channels, channels, kernel_size=1)
        self.o = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        x = x.view(B, C, H * W)
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn_scores = torch.bmm(q.transpose(1, 2), k) / (C ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)
        out = torch.bmm(v, attn_scores.transpose(1, 2))
        out = self.o(out)
        return out.view(B, C, H, W) + residual
    
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, embed_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, emb):
        out = self.res_block(x, emb)
        return self.pool(out), out 
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(in_channels + out_channels, out_channels, embed_dim)

    def forward(self, x, skip, emb):
        up = self.upsample(x)
        x_cat = torch.cat([up, skip], dim=1)
        return self.res_block(x_cat, emb)
    
class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        assert len(config.channels) == config.depth + 1, "Length of channels must match depth"
        assert config.image_size % (2 ** config.depth) == 0, "Image size must be divisible by 2^depth"
        assert config.embed_dim % 2 == 0, "Embedding dimension must be even"

        self.config = config
        self.in_layer = ResidualBlock(config.in_channels, config.channels[0], config.embed_dim)

        self.down_layers = nn.ModuleList()
        for i in range(config.depth):
            self.down_layers.append(DownsampleBlock(config.channels[i], config.channels[i + 1], config.embed_dim))
        self.bottleneck = nn.ModuleList([ResidualBlock(config.channels[-1], config.channels[-1], config.embed_dim) 
            for _ in range(config.bottleneck_blocks)])
        self.bottleneck_attn = AttentionBlock(config.channels[-1])
        self.up_layers = nn.ModuleList()
        for i in range(config.depth - 1, -1, -1):
            self.up_layers.append(UpsampleBlock(config.channels[i + 1], config.channels[i], config.embed_dim))
        self.out_layer = nn.Conv2d(config.channels[0], config.out_channels, kernel_size=1)

        self.res_blocks_down = nn.ModuleList()
        for i in range(config.depth):
            self.res_block = nn.ModuleList()
            for _ in range(config.res_blocks):
                self.res_block.append(ResidualBlock(config.channels[i + 1], config.channels[i + 1], config.embed_dim))
            self.res_blocks_down.append(self.res_block)

        self.res_blocks_up = nn.ModuleList()
        for i in range(config.depth - 1, -1, -1):
            self.res_block = nn.ModuleList()
            for _ in range(config.res_blocks):
                self.res_block.append(ResidualBlock(config.channels[i], config.channels[i], config.embed_dim))
            self.res_blocks_up.append(self.res_block)

        self.attn_blocks_down = nn.ModuleList()
        for i in range(config.depth):
            self.attn_blocks_down.append(AttentionBlock(config.channels[i + 1]))
        self.attn_blocks_up = nn.ModuleList()
        for i in range(config.depth - 1, -1, -1):
            self.attn_blocks_up.append(AttentionBlock(config.channels[i]))

    def forward(self, x, t):
        emb = sine_embedding(t, dim=self.config.embed_dim).to(x.dtype)

        x = self.in_layer(x, emb)
        skips = []
        for i in range(self.config.depth):
            x, skip = self.down_layers[i](x, emb)
            skips.append(skip)
            for block in self.res_blocks_down[i]:
                x = block(x, emb)
            x = self.attn_blocks_down[i](x)

        for block in self.bottleneck:
            x = block(x, emb)
        x = self.bottleneck_attn(x)

        for up_layer, res_blocks, attn_block in zip(self.up_layers, self.res_blocks_up, self.attn_blocks_up):
            skip = skips.pop()  
            x = up_layer(x, skip, emb)
            for block in res_blocks:
                x = block(x, emb)
            x = attn_block(x)

        return self.out_layer(x)
    
def sine_embedding(t, dim=32):
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb