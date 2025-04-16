import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
from model import UNet, UNetConfig
from diffusion import forward_diffusion, sample_ddim
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class TrainConfig:
    batch_size: int = 100
    learning_rate: float = 1e-4
    num_epochs: int = 100
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 1e-2  
    gradient_clip: float = 1.0  
    img_size: int = 32
    save_steps: int = None
    output_dir: str = './output_images'
    T: int = 1000
    T_sample: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader: DataLoader = None
    transform: transforms.Compose = None
    schedule: tuple = (1e-4, 0.02)

def train(config: TrainConfig, model_config: UNetConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    train_loader = config.dataloader

    beta_schedule = torch.linspace(config.schedule[0], config.schedule[1], config.T).to(config.device)
    alpha_cumprod = torch.cumprod(1 - beta_schedule, dim=0)

    model = UNet(model_config).to(config.device, dtype=torch.get_default_dtype())
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    
    losses_epochs = []
        
    for epoch in range(config.num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        epoch_loss = 0.0
        
        optimizer.zero_grad()  
        
        for batch_idx, (images, _) in enumerate(loop):
            images = images.to(config.device, dtype=torch.get_default_dtype())
            
            t = torch.randint(0, config.T, (images.shape[0],), device=config.device, dtype=torch.long)
            x_t, noise = forward_diffusion(images, t, alpha_cumprod, config.device)
            
            pred_noise = model(x_t, t)
            loss = loss_fn(pred_noise, noise) 
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()  
            loop.set_postfix(loss=loss.item())
            
            if config.save_steps is not None and (batch_idx + 1) % config.save_steps == 0:
                with torch.no_grad():
                    generated_images = sample_ddim(
                        model,
                        config.img_size, 
                        model_config.in_channels, 
                        config.T, 
                        config.T_sample,
                        config.device, 
                        config.batch_size,
                        alpha_cumprod
                    )
                    torchvision.utils.save_image(
                        generated_images, 
                        os.path.join(config.output_dir, f"step_{batch_idx}.png"), 
                        nrow=4
                    )

                    torch.save(model.state_dict(), f"model.pth")
                
        with torch.no_grad():
            generated_images = sample_ddim(
                model,
                config.img_size, 
                model_config.in_channels, 
                config.T, 
                config.T_sample,
                config.device, 
                config.batch_size,
                alpha_cumprod
            )
            torchvision.utils.save_image(
                generated_images, 
                os.path.join(config.output_dir, f"epoch_{epoch+1}.png"), 
                nrow=4
            )
        
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {epoch_loss/len(train_loader)}")
        losses_epochs.append(epoch_loss / len(train_loader))
        
        torch.save(model.state_dict(), f"model.pth")
        
        plt.figure(figsize=(10, 5))
        plt.plot(losses_epochs, label='Epoch Loss')
        plt.legend()
        plt.savefig(os.path.join(config.output_dir, 'loss_plot.png'))
        plt.close()