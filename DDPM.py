from Unet import Unet
import torch
import torch.nn as nn
from NoiseSchedule import *
import torch.nn.functional as F


class DDPM(nn.Module):
    def __init__(self, unet: nn.Module, n_steps: int, device: torch.device, noise_schedule="cosin"):
        super().__init__()
        self.eps_model = unet
        self.n_steps = n_steps
        if noise_schedule == "cosin":
            self.beta = cosin_schedule(n_steps)
        elif noise_schedule == "linear":
            self.beta = linear_schedule(n_steps)
        # use beta to get alpha and alpha_bara
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)

        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        # 按照t取出对应的alpha_bar
        alpha_bars = self.alpha_bar.gather(-1, t)
        beta_bars = 1. - alpha_bars
        alpha_bars = alpha_bars ** 0.5
        beta_bars = beta_bars ** 0.5
        # 添加到相同维度
        while len(alpha_bars.shape) < len(x0.shape):
            alpha_bars = alpha_bars.unsqueeze(-1)
            beta_bars = beta_bars.unsqueeze(-1)
        noisy_img = alpha_bars * x0 + beta_bars * noise
        return noisy_img

    def reverse(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.eps_model(xt, t)
        alpha_bar = self.alpha_bar.gather(-1, t)
        alpha = self.alpha.gather(-1, t)
        eps_t = torch.rand_like(xt, device=xt.device)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        # xt-1= [xt - beta * eps / sqrt(1 - alpha_bar)] / sqrt(alpha) + sqrt(beta) * eps_t
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.beta.gather(-1, t)
        return mean + (var ** 0.5) * eps_t

    def loss(self, x0: torch.Tensor, noise: torch.Tensor = None):
        batch_size = x0.shape[0]
        # 抽样时间t
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)
        # 添加噪声
        xt = self.add_noise(x0, t, noise)
        # 估计噪声
        eps_predict = self.eps_model(xt, t)
        # 返回损失，使用L2 Loss
        return F.mse_loss(noise, eps_predict)
