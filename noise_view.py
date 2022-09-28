import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import config
from dataset import ImageDataset
from utils import Diffusion

root_dir = config.Dataset_root
device = "cuda" if torch.cuda.is_available() else "cpu"
diffusion = Diffusion(img_size=config.Image_size,timesteps=1000 ,device=device)

dataset = ImageDataset(root_dir)
dataloader = DataLoader(
    dataset,
    batch_size=config.Batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.Num_workers,
)

for i in range(1000):
    for idx, image in enumerate(dataloader):
        image = image.to(device)
        t = torch.tensor([i]).to(device)
        print(t)
        image_noisy, noise = diffusion.noise_images(image, t)
        if i%10 == 0:
            save_image(image_noisy, f"noise_images/image_noisy_{i}_{idx}.png")
        # save_image(noise, f"noise_images/noise_{i}_{idx}.png")