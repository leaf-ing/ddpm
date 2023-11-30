from TimeEmbedding import *


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_channels, n_groups=32, dropout=0.1):
        super().__init__()
        # 2 conv
        self.conv1 = nn.Sequential(
            # self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
            # self.norm1 = nn.GroupNorm(n_groups, in_channel)
            # self.act1 = Swish()
            nn.GroupNorm(n_groups, in_channel),
            Swish(),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        )
        self.norm2 = nn.GroupNorm(n_groups, out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.act2 = Swish()
        # short cut
        if in_channel == out_channel:
            self.short_cut = nn.Identity()
        else:
            self.short_cut = nn.Conv2d(in_channel, out_channel, 1)
        # time_emb
        self.t_emb = nn.Linear(time_channels, out_channel)
        self.t_act = Swish()
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        # conv1
        h = self.conv1(x)
        # add time information
        # t:[batch,dim] h:[b,c,h,w]
        h += self.t_emb(self.t_act(t))[:, :, None, None]
        # conv2
        h = self.act2(self.norm2(h))
        h = self.conv2(self.dropout(h))
        # shortcut
        return h + self.short_cut(x)


class UpSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    # t用于封装
    def forward(self, x, t=None):
        return self.conv(x)
