import torch
from model import UNet
import config
import os
from utils import *
from torchvision.utils import save_image


def load_checkpoint(model, dir, device):
    model.load_state_dict(torch.load(dir, map_location=device))

# 设置生成的图片张数。Set the number of generated pictures.
gen_image_num = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
if os.path.exists(config.Weights_path):
    print("<========load weight========>")
    load_checkpoint(model, config.Weights_path,device)
diffusion = Diffusion(img_size=config.Image_size, timesteps=config.Noise_timesteps, device=device)

for i in range(gen_image_num):
    sampled_images = diffusion.sample(model, n=1)
    save_image(sampled_images,f"result_images/{i}.png")