from typing import Type

import torch
from torch import nn
from torch.nn import functional as F

from . import gan_loss


class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.h1_dim = 1024
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.h1_dim),
            nn.BatchNorm1d(self.h1_dim),
            nn.ReLU(inplace=True)
        )
        self.h2_nchan = 128
        h2_dim = 7 * 7 * self.h2_nchan
        self.fc2 = nn.Sequential(
            nn.Linear(self.h1_dim, h2_dim),
            nn.BatchNorm1d(h2_dim),
            nn.ReLU(inplace=True)
        )
        self.h3_nchan = 64
        self.conv1 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(self.h2_nchan, self.h3_nchan,
            #           kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose2d(self.h2_nchan, self.h3_nchan,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h3_nchan),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(self.h3_nchan, 1,
            #           kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose2d(self.h3_nchan, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x).view(-1, self.h2_nchan, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1_nchan = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.h1_nchan, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(.1, inplace=True)
        )
        self.h2_nchan = 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.h1_nchan, self.h2_nchan, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h2_nchan),
            nn.LeakyReLU(.1, inplace=True)
        )
        self.h3_dim = 1024
        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * self.h2_nchan, self.h3_dim),
            nn.BatchNorm1d(self.h3_dim),
            nn.LeakyReLU(.1, inplace=True)
        )
        self.fc2 = nn.Linear(self.h3_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x).view(-1, 7 * 7 * self.h2_nchan)
        h = self.fc1(x)
        x = self.fc2(h)
        return x, h


class Recognition(nn.Module):
    def __init__(self, h_dim: int, cat_dim: int, cont_dim: int, bin_dim: int):
        super().__init__()
        self.h1_dim = 128
        self.fc1 = nn.Sequential(
            nn.Linear(h_dim, self.h1_dim),
            nn.BatchNorm1d(self.h1_dim),
            nn.LeakyReLU(.1)
        )
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim
        self.bin_dim = bin_dim
        self.fc2_cat = nn.Linear(self.h1_dim, cat_dim) if self.cat_dim > 0 else None
        self.fc2_mean = nn.Linear(self.h1_dim, cont_dim) if self.cont_dim > 0 else None
        self.fc2_var = nn.Linear(self.h1_dim, cont_dim) if self.cont_dim > 0 else None
        self.fc2_bin = nn.Linear(self.h1_dim, bin_dim) if self.bin_dim > 0 else None

    def forward(self, x):
        x = self.fc1(x)
        cat_logit = self.fc2_cat(x) if self.cat_dim > 0 else None
        cont_mean = self.fc2_mean(x) if self.cont_dim > 0 else None
        cont_logvar = self.fc2_var(x) if self.cont_dim > 0 else None
        bin_logit = self.fc2_bin(x) if self.bin_dim > 0 else None
        return cat_logit, cont_mean, cont_logvar, bin_logit


def sample_noise(num, dim, device=None) -> torch.Tensor:
    return torch.randn(num, dim, device=device)


def sample_code(num, cat_dim=0, cont_dim=0, bin_dim=0, device=None) -> torch.Tensor:
    cat_onehot = cont = bin = None
    if cat_dim > 0:
        cat = torch.randint(cat_dim, size=(num, 1), dtype=torch.long, device=device)
        cat_onehot = torch.zeros(num, cat_dim, dtype=torch.float, device=device)
        cat_onehot.scatter_(1, cat, 1)
    if cont_dim > 0:
        cont = 2. * torch.rand(num, cont_dim, device=device) - 1.
    if bin_dim > 0:
        bin = (torch.rand(num, bin_dim, device=device) > .5).float()
    return torch.cat([x for x in [cat_onehot, cont, bin] if x is not None], 1)


class InfoGAN(nn.Module):
    def __init__(self, cat_dim: int, cont_dim: int, bin_dim: int, noise_dim: int):
        super().__init__()
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim
        self.bin_dim = bin_dim
        self.noise_dim = noise_dim

        splits = [self.cat_dim, self.cont_dim, self.bin_dim, self.noise_dim]
        for i in range(1, len(splits)):
            splits[i] += splits[i - 1]  # Cumulative sum
        self.cat_idx = list(range(splits[0]))
        self.cont_idx = list(range(splits[0], splits[1]))
        self.bin_idx = list(range(splits[1], splits[2]))
        self.noise_idx = list(range(splits[2], splits[3]))
        self.code_idx = self.cat_idx + self.cont_idx + self.bin_idx

        latent_dim = self.cat_dim + self.cont_dim + self.bin_dim + self.noise_dim
        self.gen = Generator(latent_dim)
        self.dis = Discriminator()
        self.rec = Recognition(self.dis.h3_dim, self.cat_dim, self.cont_dim, self.bin_dim)

        self.apply(_weights_init)

    def sample_noise(self, num: int):
        return sample_noise(num, self.noise_dim, self.device)

    def sample_code(self, num: int):
        return sample_code(num, self.cat_dim, self.cont_dim, self.bin_dim, self.device)

    def sample_latent(self, num: int):
        code = sample_code(num, self.cat_dim, self.cont_dim, self.bin_dim, self.device)
        noise = sample_noise(num, self.noise_dim, self.device)
        return torch.cat([code, noise], 1)

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


def _gaussian_loss(x, mean, logvar):
    return .5 * torch.mean(torch.sum((x - mean) ** 2 / torch.exp(logvar) + logvar, dim=1))


class Trainer(nn.Module):
    def __init__(self, model: InfoGAN, loss_type: Type[gan_loss.GANLossType] = gan_loss.MinimaxLoss,
                 penalty: gan_loss.GradientPenaltyBase = None, info_weight: float = 1.,
                 gen_lr: float = 1e-3, dis_lr: float = 2e-4, rec_lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.info_weight = info_weight

        self.loss_type = loss_type
        self.penalty = penalty

        betas = (.5, .99)
        self.gen_opt = torch.optim.Adam(self.model.gen.parameters(), lr=gen_lr, betas=betas)
        self.dis_opt = torch.optim.Adam(self.model.dis.parameters(), lr=dis_lr, betas=betas)
        self.rec_opt = torch.optim.Adam(self.model.rec.parameters(), lr=rec_lr, betas=betas)

    @staticmethod
    def _information_loss(model: InfoGAN, fake_hidden, latent):
        cat_logit, cont_mean, cont_logvar, bin_logit = model.rec(fake_hidden)
        info_loss = 0.
        if model.cat_dim > 0:
            cat_code = latent[:, model.cat_idx]
            info_loss += F.cross_entropy(cat_logit, cat_code.argmax(1))
        if model.cont_dim > 0:
            cont_code = latent[:, model.cont_idx]
            info_loss += .1 * _gaussian_loss(cont_code, cont_mean, cont_logvar)
        if model.bin_dim > 0:
            bin_code = latent[:, model.bin_idx]
            info_loss += 2 * F.binary_cross_entropy_with_logits(bin_logit, bin_code)
        return info_loss

    def step(self, real_data, verbose: bool = False):
        batch_size = real_data.shape[0]

        real_dis_logit, real_hidden = self.model.dis(real_data)

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

        info_loss = self._information_loss(self.model, fake_hidden, latent)  # type: torch.Tensor
        info_loss *= self.info_weight

        self.gen_opt.zero_grad()
        self.dis_opt.zero_grad()
        self.rec_opt.zero_grad()
        info_loss.backward()
        self.gen_opt.step()
        self.dis_opt.step()
        self.rec_opt.step()

        if verbose:
            real_dis = F.sigmoid(real_dis_logit)
            fake_dis = F.sigmoid(fake_dis_logit)
            text = (f"D_loss = {dis_loss.item():.4f}, "
                    f"G_loss = {gen_loss.item():.4f}, "
                    f"MI = {info_loss.item():.4f}, "
                    f"D(x) = {real_dis.mean().item():.4f}, "
                    f"D(G(z)) = {fake_dis.mean().item():.4f}")
            if self.penalty is not None:
                text += f", |grad D| = {grad_norm.item():.4f}"
            print(text)

    def forward(self, real_data, verbose: bool = False):
        self.step(real_data, verbose)
