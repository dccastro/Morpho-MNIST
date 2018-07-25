import math

import torch
import torch.nn.functional as F
from torch import autograd


class GANLossType(object):
    @staticmethod
    def discriminator_loss(real_logit, fake_logit):
        raise NotImplementedError

    @staticmethod
    def generator_loss(fake_logit):
        raise NotImplementedError


class MinimaxLoss(GANLossType):
    @staticmethod
    def discriminator_loss(real_logit, fake_logit):
        return -F.logsigmoid(real_logit).mean() - F.logsigmoid(-fake_logit).mean()

    @staticmethod
    def generator_loss(fake_logit):
        return F.logsigmoid(-fake_logit).mean()


class NonSaturatingLoss(MinimaxLoss):
    @staticmethod
    def generator_loss(fake_logit):
        return -F.logsigmoid(fake_logit).mean()


class LeastSquaresLoss(GANLossType):
    @staticmethod
    def discriminator_loss(real_logit, fake_logit):
        return ((real_logit - 1.) ** 2).mean() + (fake_logit ** 2).mean()

    @staticmethod
    def generator_loss(fake_logit):
        return ((fake_logit - 1.) ** 2).mean()


class WassersteinLoss(GANLossType):
    @staticmethod
    def discriminator_loss(real_logit, fake_logit):
        return -real_logit.mean() + fake_logit.mean()

    @staticmethod
    def generator_loss(fake_logit):
        return -fake_logit.mean()


class GradientPenaltyBase(object):
    def __init__(self, weight, target=1.):
        self.weight = weight
        self.target = target

    def get_probe(self, real_data, fake_data):
        alpha = torch.rand([real_data.shape[0]] + [1] * len(real_data.shape[1:]),
                           device=real_data.device)
        return alpha * real_data + (1. - alpha) * fake_data

    def penalty(self, dis, real_data, fake_data):
        probe = self.get_probe(real_data.detach(), fake_data.detach())
        probe.requires_grad = True
        probe_logit, _ = dis(probe)
        gradients = autograd.grad(outputs=F.sigmoid(probe_logit),
                                  inputs=probe,
                                  grad_outputs=torch.ones_like(probe_logit))[0]
        grad_norm = gradients.view(gradients.shape[0], -1).norm(2, dim=1)
        penalty = ((grad_norm - self.target) ** 2).mean()
        return self.weight * penalty, grad_norm.mean()


class DeepRegretPenalty(GradientPenaltyBase):
    def __init__(self, variance, weight, target=1.):
        super().__init__(weight, target)
        self.scale = math.sqrt(variance)

    def get_probe(self, real_data, fake_data):
        return real_data + self.scale * torch.randn_like(real_data)
