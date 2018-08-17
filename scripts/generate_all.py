import multiprocessing
import os
import shutil

import numpy as np

from morphomnist import idx, perturb
from morphomnist.morpho import ImageMorphology

THRESHOLD = .5
UP_FACTOR = 4

PERTURBATIONS = [
    perturb.Thinning(.7),
    perturb.Thickening(1.),
    perturb.Swelling(3, 7),
    perturb.Fracture(num_frac=3)
]


def process_image(args):
    i, img = args
    np.random.seed()
    morph = ImageMorphology(img, THRESHOLD, UP_FACTOR)
    out_imgs = [morph.downscale(morph.binary_image)] + \
               [morph.downscale(pert(morph)) for pert in PERTURBATIONS]
    return out_imgs


if __name__ == '__main__':
    raw_dir = "/vol/biomedic/users/dc315/mnist/raw"
    dataset_root = "/vol/biomedic/users/dc315/mnist_new"
    dataset_names = ["plain", "thin", "thic", "swel", "frac"]

    pool = multiprocessing.Pool()
    for subset in ["train", "t10k"]:
        imgs_filename = f"{subset}-images-idx3-ubyte.gz"
        labels_filename = f"{subset}-labels-idx1-ubyte.gz"
        raw_imgs = idx.load(os.path.join(raw_dir, imgs_filename))

        gen = pool.imap(process_image, enumerate(raw_imgs), chunksize=100)
        try:
            import tqdm
            gen = tqdm.tqdm(gen, total=len(raw_imgs), unit='img', ascii=True)
        except ImportError:
            def plain_progress(g):
                print(f"\rProcessing images: 0/{len(raw_imgs)}", end='')
                for i, res in enumerate(g):
                    print(f"\rProcessing images: {i + 1}/{len(raw_imgs)}", end='')
                    yield res
                print()
            gen = plain_progress(gen)

        result = zip(*list(gen))
        for dataset_name, imgs in zip(dataset_names, result):
            imgs = np.array(imgs)
            dataset_dir = os.path.join(dataset_root, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            idx.save(imgs, os.path.join(dataset_dir, imgs_filename))
            shutil.copy(os.path.join(raw_dir, labels_filename), dataset_dir)
    pool.close()
    pool.join()
