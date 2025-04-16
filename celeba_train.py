import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import UNetConfig
from train import TrainConfig, train
import matplotlib.pyplot as plt

batch_size = 10
img_size = 64

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

train_dataset = ImageFolder(root='data/celeba/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

def display_images(images):
    for i in range(len(images)):
        images[i] = (images[i] + 1) / 2
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
    plt.savefig('sample.png')
    
images, _ = next(iter(train_loader))
display_images(images)

train_config = TrainConfig(
    batch_size=batch_size,
    learning_rate=1e-4,
    num_epochs=100,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    gradient_clip=1.0,
    img_size=img_size,
    save_steps=None,
    output_dir='./output_images',
    T=1000,
    T_sample=100,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    dataloader=train_loader,
    transform=transform,
    schedule=(1e-4, 0.02),
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

train(
    model_config=model_config,
    config=train_config,
)