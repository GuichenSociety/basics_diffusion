import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils import *



class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,time_emb_dim,up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if up:
            self.conv1 = nn.Conv2d(2*in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU()

    def forward(self,x,t):
        h = self.bnorm1(self.relu(self.conv2(self.conv1(x))))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv3(h)))
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class UNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,time_emb_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)

        self.down1 = Residual(64,128,time_emb_dim)
        self.down2 = Residual(128, 256, time_emb_dim)
        self.down3 = Residual(256, 512, time_emb_dim)
        self.down4 = Residual(512, 1024, time_emb_dim)

        self.up1 = Residual(1024, 512, time_emb_dim,up=True)
        self.up2 = Residual(512, 256, time_emb_dim, up=True)
        self.up3 = Residual(256, 128, time_emb_dim, up=True)
        self.up4 = Residual(128, 64, time_emb_dim, up=True)

        self.conv1 = nn.Conv2d(64*2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.output = nn.Conv2d(16, out_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)

        x1 = self.down1(x,t)
        x2 = self.down2(x1, t)
        x3 = self.down3(x2, t)
        x4 = self.down4(x3, t)

        x5 = torch.cat((x4, x4), dim=1)
        x5 = self.up1(x5,t)
        x6 = torch.cat((x5, x3), dim=1)
        x6 = self.up2(x6,t)
        x7 = torch.cat((x6, x2), dim=1)
        x7 = self.up3(x7,t)
        x8 = torch.cat((x7, x1), dim=1)
        x8 = self.up4(x8,t)

        x = torch.cat((x8, x), dim=1)
        x = self.conv2(self.conv1(x))

        return self.output(x)




if __name__ == '__main__':
    u = UNet()
    print(u)
