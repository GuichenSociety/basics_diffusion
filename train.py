import torch
import os
import config
from tqdm import tqdm
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from model import UNet
from utils import *



torch.backends.cudnn.benchmark = True

class train:
    def __init__(self,root_dir,device,timesteps = 1000,):

        self.timesteps = timesteps

        self.diffusion = Diffusion(img_size=config.Image_size,timesteps=timesteps ,device=device)

        self.device = device
        dataset = ImageDataset(root_dir)
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.Batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.Num_workers,
        )

        self.model = UNet().to(device)
        self.optimizer = Adam(self.model.parameters(), lr=config.Learning_rate, betas=(0.9, 0.99))
        self.mse = nn.MSELoss()

        if os.path.exists(config.Weights_path):
            print("<========load weight========>")
            self.load_checkpoint(self.model,config.Weights_path,self.device)

    def load_checkpoint(self,model,dir,device):
        model.load_state_dict(torch.load(dir, map_location=device))

    def save_checkpoint(self, model, filename=config.Weights_path):
        # print("=> Saving checkpoint")
        checkpoint = model.state_dict()
        torch.save(checkpoint, filename)

    def epochs(self,epoch,epochs):
        loop = tqdm(self.dataloader, leave=True)
        for idx, image in enumerate(loop):
            image = image.to(self.device)
            t = self.diffusion.sample_timesteps(image.shape[0])

            image_noisy, noise = self.diffusion.noise_images(image, t)
            noise_pred = self.model(image_noisy, t)
            loss = self.mse(noise_pred,noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                loop.set_postfix(loss=loss.item())
                # loop.set_postfix(step=f"{int(epoch/epochs*100)}%")

        if epoch % 10 == 0:
            self.save_checkpoint(self.model)
            sampled_images = self.diffusion.sample(self.model, n=config.Batch_size)
            save_image(sampled_images,f"gen_images1/{epoch}_{idx}.png")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = train(root_dir=config.Dataset_root,device=device,timesteps =config.Noise_timesteps)
    for epoch in range(config.Num_epochs):
        train.epochs(epoch,config.Num_epochs)
