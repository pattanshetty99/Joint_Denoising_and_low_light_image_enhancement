import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerBlock


class Encoder(nn.Module):
    def __init__(self, in_ch=3, dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)

        self.t1 = TransformerBlock(dim)
        self.t2 = TransformerBlock(dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.t1(x)
        x = F.relu(self.conv3(x))
        x = self.t2(x)
        return x


class StructureDecoder(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
        self.conv = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        return torch.sigmoid(self.conv(x))


class ColorBranch(nn.Module):
    def __init__(self, in_ch=3, dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 3, 3, padding=1)
        )

    def forward(self, x):
        return 0.3 * self.net(x)


class DenoiseBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

    def forward(self, x):
        return x - self.net(x)


class IlluminationBranch(nn.Module):
    def __init__(self, in_ch=3, dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
