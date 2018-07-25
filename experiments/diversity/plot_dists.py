import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.diversity import dist_plots
from experiments.diversity.measure_samples import METRICS_ROOT

CHECKPOINT_ROOT = "/data/morphomnist/checkpoints"
FIGURE_ROOT = "/vol/biomedic/users/dc315/morphomnist/fig"


def main(metrics, figure_path=None):
    # cols = ['length', 'thickness', 'slant', 'aspect']
    # lims = [(10, 70), (0, 7), (-45, 45), (0, 2.5)]
    cols = ['length', 'thickness', 'slant', 'width', 'height']
    lims = [(10, 70), (0, 7), (-45, 45), (0, 22), (8, 22)]
    multiples = {'length': 20, 'thickness': 2, 'slant': 30, 'width': 10, 'height': 5, 'aspect': .5}
    formats = {'slant': "%d\u00b0", 'aspect': "%g"}
    metrics['slant'] = np.rad2deg(metrics['slant'])

    dist_plots.plot_distribution(metrics, cols, lims, multiples, formats)
    if figure_path is not None:
        fig_kwargs = dict(dpi=400, bbox_inches='tight', pad_inches=0)
        plt.savefig(figure_path, **fig_kwargs)
    plt.show()


if __name__ == '__main__':
    dataset_dir = "/vol/biomedic/users/dc315/mnist/plain"
    test_metrics = pd.read_csv(os.path.join(dataset_dir, "t10k-morpho.csv"))
    figure_path = os.path.join(FIGURE_ROOT, f"test_plain_morpho_dist.pdf")

    specs = [
        "VAE-64_plain",
        "GAN-64_plain",
        "GAN-2_plain",
        "GAN-1_plain",
    ]

    pool = multiprocessing.Pool()
    pool.apply_async(main, (test_metrics, figure_path))
    for spec in specs:
        sample_metrics = pd.read_csv(os.path.join(METRICS_ROOT, f"{spec}_metrics.csv"))
        figure_path = os.path.join(FIGURE_ROOT, f"{spec}_morpho_dist.pdf")
        pool.apply_async(main, (sample_metrics, figure_path))
    pool.close()
    pool.join()
