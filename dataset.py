import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms

import config


class ImageDataset(Dataset):
    def __init__(self,image_paths="F:/datasets/anime_1024x1024/images"):
        super().__init__()

        self.image_paths = image_paths
        self.images_name = os.listdir(self.image_paths)

        print(f"加载了{len(self.images_name)}张图片")


        self.transform = transforms.Compose([
            transforms.Resize(config.Image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])


    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_paths,self.images_name[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image



if __name__ == '__main__':
    im = ImageDataset()
    print(im[1])