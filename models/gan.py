from typing import Type

import torch
from torch import nn
from torch.nn import functional as F

from . import gan_loss
from .infogan import Generator, Discriminator


class GAN(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.gen = Generator(self.latent_dim)
        self.dis = Discriminator()

        self.apply(_weights_init)

    def sample_latent(self, num: int):
        return torch.randn(num, self.latent_dim, device=self.device)

    def forward(self, num: int = 1):
        latent = self.sample_latent(num)
        fake_data = self.gen(latent)
        return fake_data

    @property
    def device(self):
        return next(self.parameters()).device


def _weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)


class Trainer(nn.Module):
    def __init__(self, model: GAN,
                 loss_type: Type[gan_loss.GANLossType] = gan_loss.NonSaturatingLoss,
                 penalty: gan_loss.GradientPenaltyBase = None,
                 gen_lr: float = 1e-3, dis_lr: float = 2e-4):
        super().__init__()
        self.model = model

        self.loss_type = loss_type
        self.penalty = penalty

        betas = (.5, .99)
        self.gen_opt = torch.optim.Adam(self.model.gen.parameters(), lr=gen_lr, betas=betas)
        self.dis_opt = torch.optim.Adam(self.model.dis.parameters(), lr=dis_lr, betas=betas)

    def step(self, real_data, verbose: bool = False):
        batch_size = real_data.shape[0]

        real_dis_logit, _ = self.model.dis(real_data)

        latent = self.model.sample_latent(batch_size)

        fake_data = self.model.gen(latent)
        fake_dis_logit, fake_hidden = self.model.dis(fake_data.detach())
        dis_loss = self.loss_type.discriminator_loss(real_dis_logit, fake_dis_logit)
        if self.penalty is not None:
            dis_penalty, grad_norm = self.penalty.penalty(self.model.dis, real_data, fake_data)
        else:
            dis_penalty = 0.
            grad_norm = None

        self.dis_opt.zero_grad()
        (dis_loss + dis_penalty).backward(retain_graph=True)
        self.dis_opt.step()

        fake_dis_logit, fake_hidden = self.model.dis(fake_data)
        gen_loss = self.loss_type.generator_loss(fake_dis_logit)

        self.gen_opt.zero_grad()
        gen_loss.backward(retain_graph=True)
        self.gen_opt.step()

        if verbose:
            real_dis = F.sigmoid(real_dis_logit)
            fake_dis = F.sigmoid(fake_dis_logit)
            text = (f"D_loss = {dis_loss.item():.4f}, "
                    f"G_loss = {gen_loss.item():.4f}, "
                    f"D(x) = {real_dis.mean().item():.4f}, "
                    f"D(G(z)) = {fake_dis.mean().item():.4f}")
            if self.penalty is not None:
                text += f", |grad D| = {grad_norm.item():.4f}"
            print(text)

    def forward(self, real_data, verbose: bool = False):
        self.step(real_data, verbose)
