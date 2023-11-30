import torch
import torch.nn as nn
from einops import rearrange


class AttentionBlock(nn.Module):
    def __init__(self, n_channels, n_head=8, n_groups=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_head = n_head
        self.d_k = n_channels // n_head
        self.scale = self.d_k ** -0.5
        # proj_in
        self.norm1 = nn.GroupNorm(n_groups, n_channels)
        self.proj_in = nn.Conv2d(n_channels, n_channels, 1)
        # q,k,v
        self.to_q = nn.Linear(n_channels, n_channels)
        self.to_v = nn.Linear(n_channels, n_channels)
        self.to_k = nn.Linear(n_channels, n_channels)
        # proj_out
        self.to_out = nn.Linear(n_channels, n_channels)

    def forward(self, x, t=None):
        # x:[b,c,h,w]
        b, c, h, w = x.shape
        # proj_in and rearange
        # x:[b,c,h,w] -> [b,seq_len,c]
        x1 = self.proj_in(self.norm1(x))
        x1 = rearrange(x1, "b c h w -> b (h w) c")
        # qkv [b,seq_len,c]
        q = self.to_q(x1)
        k = self.to_k(x1)
        v = self.to_v(x1)
        # multi head
        # [b*n_head,seq_len,c//n_head]
        q = rearrange(q, "b l (n c) -> (b n) l c", n=self.n_head)
        k = rearrange(k, "b l (n c) -> (b n) l c", n=self.n_head)
        v = rearrange(v, "b l (n c) -> (b n) l c", n=self.n_head)
        # attn
        # QK'/sqrt(dk)
        attn = torch.einsum("bic,bjc->bij", q, k) * self.scale
        # softmax
        score = torch.softmax(attn, dim=-1)
        # result & rearrange
        res = torch.einsum("bij,bjc->bic", score, v)
        res = res.view(b, -1, self.n_channels)
        res = self.to_out(res)
        # rearrange
        # [b,h*w,c]->[b,c,h,w]
        res = res.transpose(1, 2).view(b, c, h, w)
        # residual
        return res + x
