import torch
import torchvision
import os
import imageio
from model import UNet, UNetConfig
from diffusion import sample_ddim
from dataclasses import dataclass

@dataclass
class TestConfig:
    num_samples: int = 100
    T: int = 1000
    T_sample: int = 100
    output_path: str = './test_samples.png'
    gif_path: str = './diffusion_process.gif'
    model_path: str = './model.pth'
    schedule: tuple = (1e-4, 0.02)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    gif: bool = False

def test(config: TestConfig, model_config: UNetConfig):
    beta_schedule = torch.linspace(config.schedule[0], config.schedule[1], config.T).to(config.device)
    alpha_cumprod = torch.cumprod(1 - beta_schedule, dim=0)
    model = UNet(model_config).to(config.device)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    with torch.no_grad():
        generated_images, frames = sample_ddim(
            model, model_config.image_size, model_config.in_channels,
            config.T, config.T_sample, config.device, config.num_samples,
            alpha_cumprod, prog_bar=True, gif=config.gif
        )

        torchvision.utils.save_image(
            generated_images,
            config.output_path,
            nrow=max(i for i in range(int(config.num_samples**0.5), 0, -1) if config.num_samples % i == 0)
        )
        print(f"Generated images saved to {config.output_path}")

        if config.gif and frames:
            imageio.mimsave(config.gif_path, frames, fps=50)
            print(f"GIF of diffusion process saved to {config.gif_path}")
