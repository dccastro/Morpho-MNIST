import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from skimage import transform

from morphomnist.morpho import ImageMorphology
from morphomnist.perturb import op_power, op_thicken, op_thin
from morphomnist.util import plot_digit, plot_ellipse

DATA_ROOT = "../data/mnist"
THRESHOLD = 128
UP_FACTOR = 4


if __name__ == '__main__':
    transf = torchvision.transforms.ToTensor()

    train_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, transform=transf)
    test_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, transform=transf)

    ops = [op_thin, op_thicken, op_power, op_power]
    op_args = [(int(.8 * UP_FACTOR),),
               (int(.8 * UP_FACTOR),),
               (5, np.array([14, 14]) * UP_FACTOR, 4 * UP_FACTOR),
               (.2, np.array([14, 14]) * UP_FACTOR, 4 * UP_FACTOR)]
    op_names = ["Thinned", "Thickened", "Swollen", "Constricted"]

    for n in torch.randperm(len(train_set.train_data)):
        start = timeit.default_timer()
        img = train_set.train_data[n].squeeze().numpy()
        morph = ImageMorphology(img, THRESHOLD, UP_FACTOR)
        bin_img = morph.binary_image
        skel = morph.skeleton
        end = timeit.default_timer()
        print("Preprocessing: {:.1f} ms".format(1000. * (end - start)))

        fig, axs = plt.subplots(2, len(ops) + 1, figsize=(3 * len(ops), 6))

        plot_digit(img, axs[0, 0], "Original")
        plot_digit(bin_img, axs[1, 0], "Thresholded")

        for i, (op, arg, name) in enumerate(zip(ops, op_args, op_names)):
            start = timeit.default_timer()
            if op is op_power:
                skel_idx = np.where(skel)
                centre_idx = np.random.choice(len(skel_idx[0]))
                centre = (skel_idx[1][centre_idx], skel_idx[0][centre_idx])
                radius = arg[2]  # (2. * dist_map[centre[::-1]])
                op_img = op(bin_img, arg[0], centre, radius)
                plot_digit(op_img, axs[1, i + 1], name)
                plot_digit(transform.pyramid_reduce(op_img, downscale=UP_FACTOR), axs[0, i + 1],
                           name)
                plot_ellipse(*centre, 0, radius, radius, axs[1, i + 1], ec='r', fc='None', lw=1)
            else:
                op_img = op(bin_img, *arg)
                plot_digit(op_img, axs[1, i + 1], name)
                plot_digit(transform.pyramid_reduce(op_img, downscale=UP_FACTOR), axs[0, i + 1],
                           name)
            end = timeit.default_timer()
            print("{}: {:.1f} ms".format(name, 1000. * (end - start)))

        plt.tight_layout()
        plt.show()
