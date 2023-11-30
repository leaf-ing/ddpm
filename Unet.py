import torch
import torch.nn as nn
from TimeEmbedding import TimeEmbedding, Swish
from ConvBlock import ResBlock, UpSample, DownSample
from AttentionBlock import AttentionBlock


# DownBlock,封装ResBlock和AttentionBlock
class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, t_channel, n_head=8, has_attn=False, n_group=32,
                 dropout=0.1):
        super().__init__()
        self.res = ResBlock(in_channel, out_channel, t_channel, n_group, dropout)
        if has_attn:
            self.attn = AttentionBlock(in_channel, n_head, n_group)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        h = self.res(x, t)
        h = self.attn(h)
        return h


# UpBlock，同DownBlock
class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, t_channel, n_head=8, has_attn=False, n_group=32,
                 dropout=0.1):
        super().__init__()
        # 唯一区别在于ResBlock会接受skip_connect的输入
        self.res = ResBlock(in_channel + out_channel, out_channel, t_channel, n_group, dropout)
        if has_attn:
            self.attn = AttentionBlock(in_channel, n_head, n_group)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        h = self.res(x, t)
        h = self.attn(h)
        return h


# MiddleBlock
class MiddleBlock(nn.Module):
    # Res + Attn + Res
    def __init__(self, n_channel, t_channel, n_head=8, n_group=32, dropout=0.1):
        super().__init__()
        self.res1 = ResBlock(n_channel, n_channel, t_channel, n_group, dropout)
        self.attn = AttentionBlock(n_channel, n_head, n_group)
        self.res2 = ResBlock(n_channel, n_channel, t_channel, n_group, dropout)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


# 组装UNet
class Unet(nn.Module):
    def __init__(self, img_channel=3, n_channels=64, ch_mult=[1, 2, 2, 4],
                 is_attn=[False, False, False, False], n_block=2, n_head=8, n_group=32,
                 dropout=0.1):
        super().__init__()
        # 分辨率尺度数
        n_res = len(ch_mult)
        t_channel = 4 * n_channels
        # 图片投影
        self.conv_in = nn.Conv2d(img_channel, n_channels, 3, 1, 1)
        # 时间编码
        self.t_emb = TimeEmbedding(t_channel)
        # 下采样DownBlock
        down = []
        in_channel = out_channel = n_channels
        for i in range(n_res):
            out_channel = in_channel * ch_mult[i]
            for _ in range(n_block):
                down.append(
                    DownBlock(in_channel, out_channel, t_channel, n_head, is_attn[i], n_group,
                              dropout))
                in_channel = out_channel

            if i < n_res - 1:
                down.append(DownSample(out_channel))
        self.down = nn.ModuleList(down)
        # 中间层
        self.mid = MiddleBlock(out_channel, t_channel, n_head, n_group, dropout)
        # 上采样层
        up = []
        in_channel = out_channel
        for i in reversed(range(n_res)):
            # 上采样层需要注意：DownSample所对应的Skip-connect
            # 先完成ResNet和Attention的部分
            out_channel = in_channel
            for _ in range(n_block):
                up.append(UpBlock(in_channel, out_channel, t_channel, n_head, is_attn[i], n_group,
                                  dropout))
                in_channel = out_channel
            # 此处处理DownSample/conv_in的skip-connect所对应的UpBlock
            out_channel = out_channel // ch_mult[i]
            up.append(
                UpBlock(in_channel, out_channel, t_channel, n_head, is_attn[i], n_group, dropout))
            # 上采样
            if i > 0:
                up.append(UpSample(out_channel))
            in_channel = out_channel

        self.up = nn.ModuleList(up)
        # 最后的输出层，转为img_channel
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, n_channels),
            Swish(),
            nn.Conv2d(in_channel, img_channel, 3, 1, 1)
        )
        # 个人小修改,强化skip_connect
        self.scale = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0), requires_grad=True) for _ in
             range((n_block + 1) * n_res)])

    def forward(self, x, t):
        # 编码时间
        t = self.t_emb(t)
        # conv_in
        x = self.conv_in(x)
        skip = [self.scale[0] * x]
        scale_index = 1
        # down
        for module in self.down:
            x = module(x, t)
            skip.append(self.scale[scale_index] * x)
            scale_index += 1
        # mid
        x = self.mid(x,t)
        # up
        for module in self.up:
            if isinstance(module, UpSample):
                x = module(x)
            else:
                s = skip.pop()
                # [b,c,h,w],concat in c
                x = torch.cat([s, x], dim=1)
                x = module(x, t)
        # out
        x = self.conv_out(x)
        return x
