import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from experiments import spec_util
from experiments.disentanglement import corr_plots
from experiments.disentanglement.compute_pcorrs import PCORR_ROOT

FIGURE_ROOT = "/vol/biomedic/users/dc315/morphomnist/fig"


def set_size(w, h, ax=None):
    """ w, h: width, height in inches. From: https://stackoverflow.com/a/44971177 """
    ax = plt.gca() if ax is None else ax
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def plot_partial_correlation(dims, pcorr, cols, hrule=None, ax=None):
    ax = plt.gca() if ax is None else ax
    corr_plots.hinton(pcorr, cmap='RdBu', ax=ax)

    cat_dim, cont_dim, bin_dim = dims[:3]

    div_kwargs = dict(c='.8', lw=1, zorder=-1)
    if cat_dim > 0 and (cont_dim > 0 or bin_dim > 0):  # Cat vs. cont/bin separator
        ax.axvline(cat_dim - .5, **div_kwargs)
    if cont_dim > 0 and bin_dim > 0:  # Cont vs. bin separator
        ax.axvline(cat_dim + cont_dim - .5, **div_kwargs)
    if hrule is not None:  # Horizontal separator
        ax.axhline(hrule - .5, **div_kwargs)

    # If no categorical, start numbering continuous/binary from 1
    idx_offset = int(cat_dim > 0)

    ax.set_yticks(np.arange(pcorr.shape[0]))
    ax.set_yticklabels(cols)
    ax.set_xticks(np.arange(pcorr.shape[1]))
    ax.set_xticklabels(
       [f"$c_{{1}}^{{({i + 1})}}$" for i in range(cat_dim)] +
       [f"$c_{{{i + 1 + idx_offset}}}^{{}}$" for i in range(cont_dim + bin_dim)])
    ax.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)

    def add_xlabel(x, s):
        ax.text(x, pcorr.shape[0] - .5 + .2, s, ha='center', va='top', size='small')

    cat_pos = (cat_dim - 1.) / 2.
    cont_pos = cat_dim + (cont_dim - 1.) / 2.
    bin_pos = cat_dim + cont_dim + (bin_dim - 1.) / 2.
    if cont_dim > 0 and bin_dim > 0:  # Adjust positions to avoid overlap
        cont_pos -= .1
        bin_pos += .1

    if cat_dim > 0:
        add_xlabel(cat_pos, 'categorical')
    if cont_dim > 0:
        add_xlabel(cont_pos, 'continuous')
    if bin_dim > 0:
        add_xlabel(bin_pos, 'binary')


def main(pcorr_path, figure_dir=None):
    pcorr_filename = os.path.split(pcorr_path)[-1]
    spec = pcorr_filename.split("_pcorr")[0]
    _, latent_dims, dataset_names = spec_util.parse_setup_spec(spec)

    fig_size = (8, 3)
    fig_kwargs = dict(dpi=300, bbox_inches='tight', pad_inches=0)

    with open(pcorr_path, 'rb') as f:
        payload = pickle.load(f)
    plt.figure(figsize=fig_size)

    plot_partial_correlation(latent_dims, payload['pcorr'], payload['cols'], payload['hrule'])
    os.makedirs(figure_dir, exist_ok=True)
    if figure_dir is not None:
        filename = pcorr_filename.split('.')[0] + ".pdf"
        shape = np.array(payload['pcorr'].shape)
        set_size(*(.3 * shape)[::-1])
        plt.savefig(os.path.join(figure_dir, filename), **fig_kwargs)
    plt.show()


if __name__ == '__main__':
    specs = [
        "InfoGAN-10c2g62n_plain",
        "InfoGAN-10c3g62n_plain+thin+thic",
        "InfoGAN-10c2g2b62n_plain+swel+frac",
    ]
    for spec in specs:
        for label in ['test', 'sample']:
            pcorr_path = os.path.join(PCORR_ROOT, f"{spec}_pcorr_{label}.pickle")
            main(pcorr_path, FIGURE_ROOT)
