import torch
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision

def forward_diffusion(x_0, t, alpha_cumprod, device):
    noise = torch.randn_like(x_0).to(device)
    alpha_t = torch.sqrt(alpha_cumprod[t]).view(-1, 1, 1, 1)
    beta_t = torch.sqrt(1 - alpha_cumprod[t]).view(-1, 1, 1, 1)
    x_t = alpha_t * x_0 + beta_t * noise
    return x_t, noise

def sample_ddim(model, img_size, channels, T, T_sample, device, num_samples, alpha_cumprod, prog_bar=False, gif=False):
    model.eval()
    ddim_timesteps = torch.linspace(0, T - 1, steps=T_sample, dtype=torch.float).to(torch.long).flip(0)
    x_t = torch.randn(num_samples, channels, img_size, img_size).to(device)
    pbar = tqdm(range(len(ddim_timesteps) - 1), desc="Sampling", disable=not prog_bar)
    frames = [] if gif else None

    for i in pbar:
        t_current = ddim_timesteps[i]
        t_next = ddim_timesteps[i + 1]
        t = torch.full((num_samples,), t_current, device=device, dtype=torch.long)

        alpha_t = torch.sqrt(alpha_cumprod[t]).view(-1, 1, 1, 1)
        alpha_t_next = torch.sqrt(alpha_cumprod[torch.full((num_samples,), t_next, device=device, dtype=torch.long)]).view(-1, 1, 1, 1)
        beta_t = torch.sqrt(1 - alpha_cumprod[t]).view(-1, 1, 1, 1)

        pred_noise = model(x_t, t)
        x0_pred = (x_t - beta_t * pred_noise) / alpha_t
        beta_t_next = torch.sqrt(1 - alpha_cumprod[torch.full((num_samples,), t_next, device=device, dtype=torch.long)]).view(-1, 1, 1, 1)

        x_t = alpha_t_next * x0_pred + beta_t_next * pred_noise
        x_t = torch.clamp(x_t, -1.0, 1.0)

        if gif:
            grid = torchvision.utils.make_grid((x_t + 1) / 2.0, nrow=int(num_samples ** 0.5))
            frame = TF.to_pil_image(grid.cpu())
            frames.append(frame)

    x_t = (x_t + 1) / 2.0
    return x_t, frames