import torch
import torch.nn as nn
import math


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    # TimeEmbedding
    # sin(t/base^(2i/d)) cos(t/base^(2i/d))
    def __init__(self, n_channels, base=10000):
        super().__init__()
        self.n_channels = n_channels
        self.base = base
        # Layers
        self.layers = nn.Sequential(
            nn.Linear(n_channels // 4, n_channels),
            Swish(),
            nn.Linear(n_channels, n_channels),
        )
        self.freq = self.calFreq()

    def calFreq(self):
        half_dim = self.n_channels // 8
        emb = -math.log(self.base) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * emb)
        return emb

    def forward(self, t: torch.Tensor):
        # t:[batch]
        self.freq = self.freq.to(t.device)
        emb = t[:, None] * self.freq
        # emb:[bacth,n_channels//8] , cat in dim1
        t_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        t_emb = self.layers(t_emb)
        return t_emb
