# import sys
# sys.path.append('../pycharm-debug.egg')
# import pydevd
# pydevd.settrace('155.198.198.155', port=5000, stdoutToServer=True, stderrToServer=True)

import os
from numbers import Number
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from experiments import data_util, spec_util, infogan_util
from models import gan_loss, infogan, load_checkpoint, save_checkpoint
from morphomnist.util import plot_grid


def test(model: infogan.InfoGAN, cat_mapping=None):
    model.eval()

    fake_data = model(64)
    plot_grid(fake_data.detach(), figsize=(8, 8), gridspec_kw=dict(wspace=.1, hspace=.1))
    plt.show()

    nrow = 5
    if model.cat_dim > 0:
        infogan_util.plot_cat_traversal(model, nrow, cat_mapping)
        plt.show()
    if model.cont_dim > 0:
        for c in range(model.cont_dim):
            infogan_util.plot_cont_traversal(model, c, nrow)
            plt.show()
    if model.bin_dim > 0:
        infogan_util.plot_bin_traversal(model, nrow)
        plt.show()


def get_cat_mapping(model: infogan.InfoGAN, data_loader: DataLoader):
    eye = torch.eye(10)
    confusion = torch.zeros(10, 10)
    for data, labels in data_loader:
        real_data = data.to(model.device).unsqueeze(1).float() / 255.
        cat_logits = model.rec(model.dis(real_data)[1])[0]
        confusion += eye[labels.long()].t() @ eye[cat_logits.cpu().argmax(1)]
    return confusion.argmax(0).numpy()


def main(use_cuda: bool, data_dirs: Union[str, Sequence[str]], weights: Optional[Sequence[Number]],
         ckpt_root: str, cat_dim: int, cont_dim: int, bin_dim: int, noise_dim: int,
         num_epochs: int, batch_size: int, save: bool, resume: bool, plot: bool):
    device = torch.device('cuda' if use_cuda else 'cpu')

    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    dataset_names = [os.path.split(data_dir)[-1] for data_dir in data_dirs]

    dims = (cat_dim, cont_dim, bin_dim, noise_dim)
    ckpt_name = spec_util.format_setup_spec('InfoGAN', dims, dataset_names)
    print(f"Training {ckpt_name}...")
    ckpt_dir = None if ckpt_root is None else os.path.join(ckpt_root, ckpt_name)

    train_set = data_util.get_dataset(data_dirs, weights)

    dl_kwargs = dict(num_workers=1, pin_memory=True) if use_cuda else {}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **dl_kwargs)
    num_batches = len(train_loader.dataset) // train_loader.batch_size

    info_weight = 1.
    model = infogan.InfoGAN(cat_dim, cont_dim, bin_dim, noise_dim)
    trainer = infogan.Trainer(model, gan_loss.NonSaturatingLoss, info_weight=info_weight)
    trainer.to(device)

    start_epoch = -1
    if resume:
        try:
            start_epoch = load_checkpoint(trainer, ckpt_dir)
            if plot:
                cat_mapping = get_cat_mapping(model, train_loader)
                test(model, cat_mapping)
        except ValueError:
            print(f"No checkpoint to resume from in {ckpt_dir}")
        except FileNotFoundError:
            print(f"Invalid checkpoint directory: {ckpt_dir}")
    elif save:
        if os.path.exists(ckpt_dir):
            print(f"Clearing existing checkpoints in {ckpt_dir}")
            for filename in os.listdir(ckpt_dir):
                os.remove(os.path.join(ckpt_dir, filename))

    for epoch in range(start_epoch + 1, num_epochs):
        trainer.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            verbose = batch_idx % 10 == 0
            if verbose:
                print(f"[{epoch}/{num_epochs}: {batch_idx:3d}/{num_batches:3d}] ", end='')

            real_data = data.to(device).unsqueeze(1).float() / 255.
            trainer.step(real_data, verbose)

        if save:
            save_checkpoint(trainer, ckpt_dir, epoch)

        if plot:
            cat_mapping = get_cat_mapping(model, train_loader)
            test(model, cat_mapping)


def parse_infogan_spec(spec):
    import re
    match = re.match(r"^(\d+c)?(\d+g)?(\d+b)?(\d+n)?$", spec)
    if match is None:
        raise ValueError(f"Invalid InfoGAN spec string: '{spec}'")
    groups = [match.group(i + 1) for i in range(4)]
    return tuple(0 if g is None else int(g[:-1]) for g in groups)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'],
                        help="device to use for training (default: use CUDA if available)")
    parser.add_argument('--save', action='store_true',
                        help="save training state after each training epoch")
    parser.add_argument('--resume', action='store_true',
                        help="resume training from latest checkpoint, if available")
    parser.add_argument('--checkpoint',
                        help="root directory where checkpoints are saved")
    parser.add_argument('--epochs', type=int, required=True,
                        help="total number of epochs")
    parser.add_argument('--batchsize', type=int, default=64,
                        help="training batch size (default: %(default)d)")
    parser.add_argument('--data', nargs='+',
                        required=True,
                        help=("MNIST-like data directory(ies); if more than one is given, "
                              "data will be randomly interleaved"))
    parser.add_argument('--weights', nargs='+', required=False,
                        help=("weights for randomly interleaving data directories; must be "
                              "positive of the same length as the list of directories"))
    parser.add_argument('--spec', required=True,
                        help="InfoGAN latent dimension specifications: "
                             "[<CAT>c][<CONT>g][<BIN>b][<NOISE>n]")

    # argv = None
    argv = ("--epochs 20 --data /vol/biomedic/users/dc315/mnist/plain "
            "/vol/biomedic/users/dc315/mnist/pert-swel-frac "
            "--checkpoint /data/morphomnist/checkpoints "
            "--spec 10c3g62n --save --weights 1 2").split()
    args = parser.parse_args(argv)
    print(args)

    use_cuda = (args.device == 'cuda') if args.device else torch.cuda.is_available()

    if (args.save or args.resume) and args.checkpoint is None:
        raise ValueError("Save or resume requested but no checkpoint root given")
    if args.weights is not None and len(args.weights) != len(args.data):
        raise ValueError(
            f"Wrong number of weights: expected {len(args.data)}, got {len(args.weights)}")

    cat_dim, cont_dim, bin_dim, noise_dim = parse_infogan_spec(args.spec)

    main(use_cuda=use_cuda, data_dirs=args.data, ckpt_root=args.checkpoint,
         cat_dim=cat_dim, cont_dim=cont_dim, bin_dim=bin_dim, noise_dim=noise_dim,
         num_epochs=args.epochs, batch_size=args.batchsize, save=args.save, resume=args.resume,
         plot=True, weights=args.weights)
