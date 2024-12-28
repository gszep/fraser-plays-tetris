import torch
import torch.nn as nn
import torch.nn.functional as F
from tetris_school.layers import ConvolutionalBlock, Head


class Fraser(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int = 11, num_states=5, num_actions: int = 4):
        super().__init__()

        self.kernel_size = kernel_size
        self.layer_number = kernel_size // 2
        self.num_states = num_states

        self.embed = nn.Embedding(num_states, embed_dim, padding_idx=0)
        self.layers = nn.ModuleList([ConvolutionalBlock(embed_dim) for i in range(self.layer_number)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = Head(embed_dim, num_actions=num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.state_transform(x)

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x)

    def state_transform(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = F.pad(
            x, (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2), value=self.num_states - 1
        )

        input = torch.zeros((batch_size, self.kernel_size, self.kernel_size), device=x.device, dtype=x.dtype)
        batch_idx, width_idx, height_idx = torch.where(x == 3)

        for n, i, j in zip(batch_idx, width_idx, height_idx):
            patch = x[
                n,
                i - self.kernel_size // 2 : i + self.kernel_size // 2 + 1,
                j - self.kernel_size // 2 : j + self.kernel_size // 2 + 1,
            ]

            input[n] = patch

        return input


class Jordan(nn.Module):
    def __init__(self, hidden_size: int, layer_number: int, input_size: tuple = (5, 5), num_actions: int = 4):
        super().__init__()

        self.input_size = 2 * input_size[0]
        self.embed = nn.Linear(self.input_size, hidden_size)
        self.layers = nn.ModuleList([ConvolutionalBlock(hidden_size) for i in range(layer_number)])

        self.norm = nn.LayerNorm(hidden_size)
        self.head = Head(hidden_size, num_actions=num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.state_transform(x)

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x)

    def state_transform(self, x: torch.Tensor) -> torch.Tensor:
        height_range = torch.arange(x.size(2)).to(x.device)

        floor = (height_range * (x == 1)).argmax(dim=-1)
        ceiling = (height_range * (x == 2)).argmax(dim=-1)

        x = torch.cat([floor, ceiling], dim=-1)
        return x.float()
