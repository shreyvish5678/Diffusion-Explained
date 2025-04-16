from test import test, TestConfig
from model import UNetConfig
import torch

img_size = 64   

config = TestConfig(
    num_samples=1,
    T=1000,
    T_sample=100,
    output_path='./test_samples.png',
    gif_path='./diffusion_process.gif',
    model_path='./model_step_9999.pth',
    schedule=(1e-4, 0.02),
    device='cuda' if torch.cuda.is_available() else 'cpu',
    gif=True,
)

model_config = UNetConfig(
    bottleneck_blocks=4,
    res_blocks=4,
    depth=3,
    image_size=img_size,
    in_channels=3, 
    out_channels=3,  
    channels=[32, 64, 128, 256],
    embed_dim=256,
)

test(config, model_config)