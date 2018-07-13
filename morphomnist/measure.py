import multiprocessing

import numpy as np
import pandas as pd

from .morpho import bounding_parallelogram, ImageMoments, ImageMorphology


def measure_image(img: np.ndarray, threshold: float = .5, scale: int = 4, bound_frac: float = .02,
                  verbose=True):
    morph = ImageMorphology(img, threshold, scale)
    moments = ImageMoments(morph.hires_image)
    mean_thck = morph.mean_thickness
    area = morph.area
    length = morph.stroke_length
    slant = np.arctan(-moments.horizontal_shear)

    corners = bounding_parallelogram(morph.hires_image, bound_frac, moments)
    width = (corners[1][0] - corners[0][0]) / morph.scale
    height = (corners[-1][1] - corners[0][1]) / morph.scale

    if verbose:
        print(f"Thickness: {mean_thck:.2f}")
        print(f"Length: {length:.1f}")
        print(f"Slant: {np.rad2deg(slant):.0f}°")
        print(f"Dimensions: {width:.1f} x {height:.1f}")

    return area, length, mean_thck, slant, width, height


def _measure_image_unpack(arg):
    return measure_image(*arg)


def measure_batch(images: np.ndarray, threshold: float = .5, scale: int = 4,
                  bound_frac: float = .02, pool: multiprocessing.Pool = None, chunksize: int = 100):
    args = ((img, threshold, scale, bound_frac, False) for img in images)
    if pool is None:
        gen = map(_measure_image_unpack, args)
    else:
        gen = pool.imap(_measure_image_unpack, args, chunksize=chunksize)

    try:
        import tqdm
        gen = tqdm.tqdm(gen, total=len(images), unit='img', ascii=True)
    except ImportError:
        def plain_progress(g):
            i = 0
            print(f"\rProcessing images: {0}/{len(images)}", end='')
            for res in g:
                i += 1
                print(f"\rProcessing images: {i + 1}/{len(images)}", end='')
                yield res
            print()
        gen = plain_progress(gen)

    results = list(gen)
    columns = ['area', 'length', 'thickness', 'slant', 'width', 'height']
    df = pd.DataFrame(results, columns=columns)
    return df
