import torch
import math
import numpy as np

# linear schedule
def linear_schedule(n_steps):
    beta = torch.linspace(0.0001, 0.02, n_steps)
    return beta


# cosin means the alpha_bar is decay as cosin function
def cosin_schedule(n_steps, s=0.008):
    # calulate alpha_bar
    time_steps = n_steps + 1
    t = torch.linspace(0, n_steps, time_steps)
    alpha_bar = torch.cos((t / n_steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = 1. - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(beta, 0, 0.999)
