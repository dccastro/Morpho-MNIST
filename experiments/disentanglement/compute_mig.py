import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
import torch

from analysis import mig
from experiments import spec_util
from models import infogan, load_checkpoint
from morphomnist import io, measure

DATA_ROOT = "/vol/biomedic/users/dc315/mnist"
CHECKPOINT_ROOT = "/data/morphomnist/checkpoints"
MIG_ROOT = "/data/morphomnist/mig"
SPEC_TO_DATASET = {"plain": "plain",
                   "plain+thin+thic": "global",
                   "plain+swel+frac": "local"}


def encode(gan: infogan.InfoGAN, x):
    with torch.no_grad():
        _, hidden = gan.dis(x)
        cat_logits, cont_mean, cont_logvar, bin_logit = gan.rec(hidden)
    return cat_logits, cont_mean, cont_logvar, bin_logit


def interleave(arrays, which):
    for a in arrays:
        a[0] = a[0].copy()
    for i in range(1, max(which) + 1):
        idx = (which == i)
        for a in arrays:
            a[0][idx] = a[i][idx]
    return [a[0] for a in arrays]


def load_test_data(data_dirs, weights=None):
    metrics_paths = [os.path.join(data_dir, "t10k-morpho.csv") for data_dir in data_dirs]
    images_paths = [os.path.join(data_dir, "t10k-images-idx3-ubyte.gz") for data_dir in data_dirs]
    labels_paths = [os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz") for data_dir in data_dirs]
    metrics = list(map(pd.read_csv, metrics_paths))
    images = list(map(io.load_idx, images_paths))
    labels = list(map(io.load_idx, labels_paths))
    if len(data_dirs) > 1:
        if weights is not None:
            weights = np.array(weights) / np.sum(weights)
        which = np.random.choice(len(data_dirs), size=len(metrics[0]), p=weights)
        metrics, images, labels = interleave([metrics, images, labels], which)
        return metrics, images, labels, which
    else:
        return metrics[0], images[0], labels[0], None


def compute_mig(gan: infogan.InfoGAN, images, metrics, cols):
    cat_logits, mean, logvar, bin_logits = encode(gan, images)
    phi = torch.softmax(cat_logits.cpu(), dim=1).numpy()
    mu = mean.cpu().numpy()
    gamma = torch.sigmoid(bin_logits.cpu()).numpy() \
        if bin_logits is not None else np.empty([metrics.shape[0], 0])

    phi_ = phi.argmax(1)
    gamma_ = gamma > .5

    codes = np.column_stack([phi_, mu, gamma_])
    factors = metrics[cols].values
    discretize_codes = [False] + [True] * mu.shape[1] + [False] * gamma_.shape[1]
    mig_score, mi, entropy = mig.mig(codes, factors, discretize_codes=discretize_codes, bins='auto')

    print(mi / entropy)
    print("MIG:", mig_score)
    return mig_score, mi, entropy


def add_swel_frac(data_dir, metrics):
    test_pert = io.load_idx(os.path.join(data_dir, "t10k-pert-idx1-ubyte.gz"))
    metrics['swel'] = (test_pert == 3).astype(int)
    metrics['frac'] = (test_pert == 4).astype(int)


def process(gan: infogan.InfoGAN, data, metrics, cols, pcorr_dir, spec, label, hrule=None):
    mig_score, mi, entropy = compute_mig(gan, data, metrics, cols)

    payload = {
        'cols': cols,
        'hrule': hrule,
        'mig': mig_score,
        'mi': mi,
        'entropy': entropy
    }
    filename = f"{spec}_mig_{label}.pickle"
    path = os.path.join(pcorr_dir, filename)
    print("Saving output to", path)
    with open(path, 'wb') as f:
        pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)


def main(checkpoint_dir, mig_dir=None):
    spec = os.path.split(checkpoint_dir)[-1]
    _, latent_dims, dataset_names = spec_util.parse_setup_spec(spec)

    device = torch.device('cuda')
    gan = infogan.InfoGAN(*latent_dims)
    trainer = infogan.Trainer(gan).to(device)
    load_checkpoint(trainer, checkpoint_dir)
    gan.eval()

    dataset_name = SPEC_TO_DATASET['+'.join(dataset_names)]
    data_dirs = [os.path.join(DATA_ROOT, dataset_name)]
    test_metrics, test_images, test_labels, test_which = load_test_data(data_dirs)

    print(test_metrics.head())

    idx = np.random.permutation(10000)#[:1000]
    X = torch.from_numpy(test_images[idx]).float().unsqueeze(1).to(device) / 255.

    cols = ['length', 'thickness', 'slant', 'width', 'height']
    test_cols = cols[:]
    test_hrule = None
    if 'swel+frac' in spec:
        add_swel_frac(data_dirs[0], test_metrics)
        test_cols += ['swel', 'frac']
        test_hrule = len(cols)

    if mig_dir is None:
        mig_dir = checkpoint_dir
    os.makedirs(mig_dir, exist_ok=True)

    process(gan, X, test_metrics.loc[idx], test_cols, mig_dir, spec, 'test', test_hrule)

    X_ = gan(10000).detach()
    with multiprocessing.Pool() as pool:
        sample_metrics = measure.measure_batch(X_.cpu().squeeze().numpy(), pool=pool)

    sample_hrule = None
    process(gan, X_, sample_metrics, cols, mig_dir, spec, 'sample', sample_hrule)


if __name__ == '__main__':
    specs = [
        "InfoGAN-10c2g62n_plain",
        "InfoGAN-10c3g62n_plain+thin+thic",
        "InfoGAN-10c2g2b62n_plain+swel+frac",
    ]
    np.set_printoptions(precision=2, linewidth=100)
    for spec in specs:
        checkpoint_dir = os.path.join(CHECKPOINT_ROOT, spec)
        main(checkpoint_dir, MIG_ROOT)
