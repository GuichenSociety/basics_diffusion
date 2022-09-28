import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
t = torch.randint(0, 1000, (32,), device=device).long()
print(t)
image_paths = "F:/datasets/anime_1024x1024/images"
images_name = os.listdir(image_paths)
print(images_name)