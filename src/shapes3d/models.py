
import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),  # 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ImageDecoder(nn.Module):
    """Decodes latent vector back to 64x64 RGB image."""
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(input_dim, 64 * 4 * 4)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),   # 64x64
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 64, 4, 4)
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.net(x)
