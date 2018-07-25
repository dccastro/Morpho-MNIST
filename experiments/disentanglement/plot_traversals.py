import os

import matplotlib.pyplot as plt
import torch

from experiments import spec_util, infogan_util
from models import load_checkpoint
from models.infogan import InfoGAN, Trainer

DATA_ROOT = "/vol/biomedic/users/dc315/mnist"
CHECKPOINT_ROOT = "/data/morphomnist/checkpoints/weighted"
FIGURE_ROOT = "/vol/biomedic/users/dc315/morphomnist/fig"


def load_gan(spec):
    _, latent_dims, dataset_names = spec_util.parse_setup_spec(spec)
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, spec)
    device = torch.device('cuda')
    gan = InfoGAN(*latent_dims)
    trainer = Trainer(gan).to(device)
    load_checkpoint(trainer, checkpoint_dir)
    gan.eval()
    return gan


def main(trav_dir):
    spec = "InfoGAN-10c2g0b62n_plain"
    gan = load_gan(spec)

    os.makedirs(trav_dir, exist_ok=True)
    fig_kwargs = dict(dpi=300, bbox_inches='tight', pad_inches=0)

    infogan_util.plot_cat_traversal(gan, 3)
    plt.savefig(os.path.join(trav_dir, spec + "_cat_trav.pdf"), **fig_kwargs)

    infogan_util.plot_cont_traversal(gan, 1, 3)
    plt.savefig(os.path.join(trav_dir, spec + "_cont_trav.pdf"), **fig_kwargs)

    # spec = "InfoGAN-10c3g0b62n_plain+pert-thin-thic"
    # gan = load_gan(spec)
    # infogan_util.plot_cont_cont_traversal(gan, 0, 2, 7)
    # plt.savefig(os.path.join(trav_dir, spec + "_cont_cont_trav.pdf"), **fig_kwargs)


if __name__ == '__main__':
    main(FIGURE_ROOT)
