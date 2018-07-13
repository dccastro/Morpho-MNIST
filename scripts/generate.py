import timeit
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np

from morphomnist import idx, perturb, util
from morphomnist.morpho import ImageMorphology

DATA_ROOT = "../data/mnist"
THRESHOLD = .5
UP_FACTOR = 4

PERTS = [
    perturb.Thinning(.7),
    perturb.Thickening(1.),
    perturb.Swelling(3, 7),
    perturb.Fracture(num_frac=3)
]


def process_image(i, img):
    np.random.seed()
    start = timeit.default_timer()
    morph = ImageMorphology(img, THRESHOLD, UP_FACTOR)
    pert_idx = np.random.choice(len(PERTS))
    pert_img_hires = PERTS[pert_idx](morph)
    pert_img = morph.downscale(pert_img_hires)
    end = timeit.default_timer()
    print(f"[{i:5d}] Preprocessing: {1000. * (end - start):.1f} ms")
    return pert_img, pert_idx


if __name__ == '__main__':
    filenames = ["train-images-idx3-ubyte", "t10k-images-idx3-ubyte"]

    pool = multiprocessing.Pool()
    for filename in filenames:
        with open(os.path.join(DATA_ROOT, "raw", filename), 'rb') as f:
            images = idx.load_uint8(f)

        pert_results = pool.starmap(process_image, zip(np.arange(len(images)), images),
                                    chunksize=1250)

        pert_images, pert_labels = zip(*pert_results)
        pert_images = np.array(pert_images)
        pert_labels = np.array(pert_labels)
        util.plot_digit(pert_images[0])
        plt.show()
        print(pert_images.shape, pert_labels.shape, np.bincount(pert_labels))
        util.save(pert_images, os.path.join(DATA_ROOT, "pert", filename) + '.gz')

        label_filename = filename.split('-')[0] + "-pert-idx1-ubyte"
        util.save(pert_labels, os.path.join(DATA_ROOT, "pert", label_filename) + '.gz')
    pool.close()
    pool.join()
