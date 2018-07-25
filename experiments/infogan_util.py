import matplotlib.pyplot as plt
import numpy as np
import torch

from models import infogan
from morphomnist.util import plot_grid

_TICK_LABEL_SIZE = 'x-large'
_VAR_LABEL_SIZE = 'xx-large'


def _prep_ax(ax):
    ax.axis('on')
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.xaxis.set_label_position('top')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines:
        ax.spines[s].set_visible(False)


def plot_cat_traversal(model: infogan.InfoGAN, nrow, cat_mapping=None):
    cat_dim = model.cat_dim
    idx = np.argsort(cat_mapping) if cat_mapping is not None else np.arange(cat_dim)
    latent = model.sample_latent(nrow).repeat(cat_dim, 1)
    latent[:, model.cat_idx] = 0
    for d in range(cat_dim):
        latent[d * nrow: (d + 1) * nrow, model.cat_idx[idx[d]]] = 1
    samples = model.gen(latent).detach()
    fig, axs = plot_grid(samples, nrow=nrow, figsize=(cat_dim, nrow),
                         gridspec_kw=dict(wspace=0, hspace=0))
    # plt.suptitle(f"$c_1$: Categorical ({cat_dim})")
    for i in [0, -1]:
        _prep_ax(axs[i, 0])
    axs[0,  0].set_xlabel('$(1)$', ha='center', va='bottom', size=_TICK_LABEL_SIZE)
    axs[-1, 0].set_xlabel(f'$({model.cat_dim})$', ha='center', va='bottom', size=_TICK_LABEL_SIZE)

    ypos = axs[0, 0].get_position().y1

    fig.text(.5, ypos, '$c_1$', ha='center', va='bottom', size=_VAR_LABEL_SIZE)


def plot_cont_traversal(model: infogan.InfoGAN, c, nrow, nstep=9):
    values = torch.linspace(-2, 2, nstep).to(model.device)
    latent = model.sample_latent(nrow).repeat(nstep, 1)
    for r in range(nrow):
        latent[r::nrow, model.cont_idx[c]] = values
    samples = model.gen(latent).detach()
    fig, axs = plot_grid(samples, nrow=nrow, figsize=(nstep, nrow),
                         gridspec_kw=dict(wspace=0, hspace=0))
    # plt.suptitle(f"$c_{{{c + 2}}}$: Continuous (-2 to 2)")

    for i in [0, -1]:
        _prep_ax(axs[i, 0])
    axs[0,  0].set_xlabel(f'${values[ 0]:+g}$', ha='center', va='bottom', size=_TICK_LABEL_SIZE)
    axs[-1, 0].set_xlabel(f'${values[-1]:+g}$', ha='center', va='bottom', size=_TICK_LABEL_SIZE)

    ypos = axs[0, 0].get_position().y1

    fig.text(.5, ypos, f'$c_{{{c + 2}}}$', ha='center', va='bottom', size=_VAR_LABEL_SIZE)


def plot_cont_cont_traversal(model: infogan.InfoGAN, c1, c2, nstep=9):
    values = torch.linspace(-1.5, 1.5, nstep).to(model.device)
    latent = model.sample_latent(1).repeat(nstep ** 2, 1)
    for s in range(nstep):
        latent[s::nstep, model.cont_idx[c2]] = values
        latent[s * nstep:(s + 1) * nstep, model.cont_idx[c1]] = values
    samples = model.gen(latent).detach()
    fig, axs = plot_grid(samples, nrow=nstep, figsize=(nstep, nstep),
                         gridspec_kw=dict(wspace=0, hspace=0))
    # plt.suptitle(rf"$c_{{{c1 + 2}}} \times c_{{{c2 + 2}}}$: Continuous (-2 to 2)")

    for i in [(0, 0), (0, -1), (-1, 0)]:
        _prep_ax(axs[i])
    axs[ 0, 0].set_xlabel(f'${values[ 0]:+g}$', ha='center', va='bottom', size=_TICK_LABEL_SIZE)
    axs[-1, 0].set_xlabel(f'${values[-1]:+g}$', ha='center', va='bottom', size=_TICK_LABEL_SIZE)
    axs[ 0, 0].set_ylabel(f'${values[ 0]:+g}$', ha='right', va='center', rotation=0, size=_TICK_LABEL_SIZE)
    axs[ 0,-1].set_ylabel(f'${values[-1]:+g}$', ha='right', va='center', rotation=0, size=_TICK_LABEL_SIZE)

    xpos = axs[ 0, 0].get_position().x0
    ypos = axs[ 0, 0].get_position().y1

    fig.text(.5, ypos, f'$c_{{{c1 + 2}}}$', ha='center', va='bottom', size=_VAR_LABEL_SIZE)
    fig.text(xpos, .5, f'$c_{{{c2 + 2}}}$', ha='right', va='center', size=_VAR_LABEL_SIZE)


def plot_bin_traversal(model: infogan.InfoGAN, nrow, ncol=5):
    latent = model.sample_latent(nrow * ncol).view(ncol, 1, nrow, -1).repeat(1, 2, 1, 1)
    bin_code = latent[..., model.bin_idx].clone()
    for b in range(model.bin_dim):
        latent[..., model.bin_idx] = bin_code
        latent[:, 0, :, model.bin_idx[b]] = 0
        latent[:, 1, :, model.bin_idx[b]] = 1
        samples = model.gen(latent.view(int(np.prod(latent.shape[:-1])), -1)).detach()
        plot_grid(samples, nrow=nrow, figsize=(2 * ncol, nrow),
                  gridspec_kw=dict(wspace=0, hspace=0))
        plt.suptitle(f"$c_{{{model.cont_dim + b + 2}}}$: Binary (columns: 0, 1)")
