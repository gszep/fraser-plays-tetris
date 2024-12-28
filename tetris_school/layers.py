import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalBlock(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.norm(x)
        x = x.permute(dims=(0, 3, 1, 2))
        x = F.gelu(self.conv1(x))
        x = self.conv2(x)
        x = x.permute(dims=(0, 2, 3, 1))

        return x  # type: ignore


class Head(nn.Module):
    def __init__(self, embed_dim: int, num_actions: int):
        super().__init__()

        self.fc = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.toaction = nn.Linear(embed_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.permute(dims=(0, 3, 1, 2))
        x = self.fc(x)
        x = x.permute(dims=(0, 2, 3, 1))
        x = F.gelu(x)
        x = self.norm(x)

        x = x.mean(dim=(1, 2))
        return self.toaction(x)  # type: ignore
