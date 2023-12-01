from Unet import Unet
import torch
from DDPM import DDPM
from typing import List
import os
import json
import numpy as np
from PIL import Image
from DataSet import CelebA
from torch.utils.data import DataLoader, Dataset


class DDPMTrainer():
    device: torch.device = torch.device("cpu")
    eps_model: Unet
    diffusion: DDPM
    img_channel: int = 3
    img_size: int = 128
    n_channel: int = 64
    n_block: int = 2
    n_head: int = 8
    n_group: int = 32
    dropout: int = 0.1
    ch_mult: List[int] = [1, 2, 2, 4]
    is_attn: List[bool] = [False, False, False, False]
    n_steps: int = 1000

    lr: float = 1e-4
    epochs: int = 10000
    batch_size: int = 64
    dataset: Dataset
    data_loader: DataLoader
    noise_schedule: str = "cosin"
    optimizer: torch.optim.Adam
    output_path: str

    def __init__(self):
        self.eps_model = Unet(img_channel=self.img_channel, n_channels=self.n_channel,
                              ch_mult=self.ch_mult, is_attn=self.is_attn,
                              n_block=self.n_block,
                              n_group=self.n_group, dropout=self.dropout).to(self.device)

        self.ddpm = DDPM(unet=self.eps_model, n_steps=self.n_steps,
                         device=self.device,
                         noise_schedule=self.noise_schedule)

        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.lr)

        self.loss = []

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=4)

    def train(self):
        for x in self.data_loader:
            x = x.to(self.device)
            self.optimizer.zero_grad()
            loss = self.ddpm.loss(x)
            loss.backward()
            self.optimizer.step()
            self.loss.append(loss.detach().cpu().item())

    def sample(self, path):
        with torch.no_grad():
            x = torch.randn([1, self.img_channel, self.img_size, self.img_size], device=self.device)
            for t_ in range(self.n_steps):
                t_ = self.n_steps - t_ - 1
                t_tensor = torch.Tensor([t_]).long().to(self.device)
                x = self.ddpm.reverse(x, t_tensor)
        x = x.cpu().squeeze(0).permute(1, 2, 0).numpy()
        x = np.clip(x, 0, 1)
        figure = x * 255
        figure = np.round(figure, 0).astype('uint8')
        im = Image.fromarray(figure)
        im.save(path)

    def run(self):
        for i in range(self.epochs):
            self.train()
            if i % 10 == 0:
                torch.save(self.eps_model.state_dict(),
                           os.path.join(self.output_path, "epoch_{}.ckpt".format(i)))
                self.sample(os.path.join(self.output_path, "epoch_{}.png".format(i)))
            # 记录loss
            with open(os.path.join(self.output_path, "loss.json"), 'w+') as f:
                f.write(json.dumps(self.loss))

    def load_state_dict(self,path):
        self.eps_model.load_state_dict(torch.load(path))

if __name__ == "main":
    dataset = CelebA(128)
    trainer = DDPMTrainer()
    trainer.set_dataset(dataset)
    trainer.run()