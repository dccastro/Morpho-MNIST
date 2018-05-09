import timeit

import matplotlib.pyplot as plt
import numpy as np
from skimage import transform

from morpho import ImageMorphology
from operations import op_power, op_thicken, op_thin
from util import plot_digit, plot_ellipse

DATA_ROOT = "../data/mnist"
THRESHOLD = 128
UP_FACTOR = 4

OPS = [op_thin, op_thicken, op_power, op_power]
OP_ARGS = [(int(.8 * UP_FACTOR),),
           (int(.8 * UP_FACTOR),),
           (5, np.array([14, 14]) * UP_FACTOR, 4 * UP_FACTOR),
           (.2, np.array([14, 14]) * UP_FACTOR, 4 * UP_FACTOR)]
OP_NAMES = ["Thinned", "Thickened", "Swollen", "Constricted"]


def process_image(img, interactive=False):
    start = timeit.default_timer()
    morph = ImageMorphology(img, THRESHOLD, UP_FACTOR)
    bin_img = morph.binary_image
    skel = morph.skeleton

    end = timeit.default_timer()
    print("Preprocessing: {:.1f} ms".format(1000. * (end - start)))

    if interactive:
        fig, axs = plt.subplots(2, 2, figsize=(6, 6))
        plot_digit(img, axs[0, 0], "Original")
        plot_digit(bin_img, axs[1, 0], "Thresholded")

    i = np.random.randint(len(OPS))
    op, arg, name = OPS[i], OP_ARGS[i], OP_NAMES[i]
    # for i, (op, arg, name) in enumerate(zip(ops, op_args, op_names)):
    start = timeit.default_timer()
    if op is op_power:
        skel_idx = np.where(skel)
        centre_idx = np.random.choice(len(skel_idx[0]))
        centre = (skel_idx[1][centre_idx], skel_idx[0][centre_idx])
        radius = arg[2]  # (2. * dist_map[centre[::-1]])
        op_img = op(bin_img, arg[0], centre, radius)
        patho_img = transform.pyramid_reduce(op_img, downscale=UP_FACTOR)  # type: np.ndarray
        if interactive:
            plot_digit(op_img, axs[1, 1], name)
            plot_digit(patho_img, axs[0, 1], name)
            plot_ellipse(*centre, 0, radius, radius, axs[1, 1], ec='r', fc='None', lw=1)
    else:
        op_img = op(bin_img, *arg)
        patho_img = transform.pyramid_reduce(op_img, downscale=UP_FACTOR)  # type: np.ndarray
        if interactive:
            plot_digit(op_img, axs[1, 1], name)
            plot_digit(patho_img, axs[0, 1], name)

    end = timeit.default_timer()
    print("{}: {:.1f} ms".format(name, 1000. * (end - start)))

    if interactive:
        plt.tight_layout()
        plt.show()

    return (255. * patho_img).astype(np.uint8), i


if __name__ == '__main__':
    import multiprocessing
    import os
    import idx

    filenames = ["train-images-idx3-ubyte", "t10k-images-idx3-ubyte"]

    pool = multiprocessing.Pool()
    for filename in filenames:
        with open(os.path.join("../data/mnist/raw", filename), 'rb') as f:
            images = idx.load_uint8(f)
        patho_results = pool.map(process_image, images, chunksize=1250)

        patho_images, patho_labels = zip(*patho_results)
        patho_images = np.array(patho_images)
        patho_labels = np.array(patho_labels)
        plot_digit(patho_images[0])
        plt.show()
        print(patho_images.shape, patho_labels.shape)

        with open(os.path.join("../data/mnist/patho", filename), 'wb') as f:
            idx.save_uint8(patho_images, f)

        label_filename = filename.split('-')[0] + "-patho-idx1-ubyte"
        with open(os.path.join("../data/mnist/patho", label_filename), 'wb') as f:
            idx.save_uint8(patho_labels, f)
    pool.close()
    pool.join()
