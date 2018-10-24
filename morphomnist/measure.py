import multiprocessing

import numpy as np
import pandas as pd

from .morpho import bounding_parallelogram, ImageMoments, ImageMorphology


def measure_image(image: np.ndarray, threshold: float = .5, scale: int = 4, bound_frac: float = .02,
                  verbose=True):
    """Computes morphometrics for a single image.

    Parameters
    ----------
    image : numpy.ndarray (28, 28)
        Input image.
    threshold : float, optional
        A relative threshold between 0 and 1. The upsampled image will be binarised at this fraction
        between its minimum and maximum values.
    scale : int, optional
        Upscaling factor for subpixel morphological analysis (>=1).
    bound_frac : float, optional
        Fraction of image mass to discard along each dimension when computing the bounding
        parallelogram.
    verbose : bool, optional
        Whether to pretty-print the estimated morphometrics.

    Returns
    -------
    area : float
        Total area/image mass.
    length : float
        Length of the estimated skeleton.
    thickness : float
        Mean thickness along the skeleton.
    slant : float
        Horizontal shear, in radians.
    width : float
        Width of the bounding parallelogram.
    height : float
        Height of the bounding parallelogram.
    """
    morph = ImageMorphology(image, threshold, scale)
    moments = ImageMoments(morph.hires_image)
    thickness = morph.mean_thickness
    area = morph.area
    length = morph.stroke_length
    slant = np.arctan(-moments.horizontal_shear)

    corners = bounding_parallelogram(morph.hires_image, bound_frac, moments)
    width = (corners[1][0] - corners[0][0]) / morph.scale
    height = (corners[-1][1] - corners[0][1]) / morph.scale

    if verbose:
        print(f"Area: {area:.1f}")
        print(f"Length: {length:.1f}")
        print(f"Thickness: {thickness:.2f}")
        print(f"Slant: {np.rad2deg(slant):.0f}°")
        print(f"Dimensions: {width:.1f} x {height:.1f}")

    return area, length, thickness, slant, width, height


def _measure_image_unpack(arg):
    return measure_image(*arg)


def measure_batch(images: np.ndarray, threshold: float = .5, scale: int = 4,
                  bound_frac: float = .02, pool: multiprocessing.Pool = None, chunksize: int = 100):
    """Computes morphometrics for a batch of images.

    Parameters
    ----------
    images : numpy.ndarray (N, 28, 28)
        Input image batch.
    threshold : float, optional
        A relative threshold between 0 and 1. The upsampled image will be binarised at this fraction
        between its minimum and maximum values.
    scale : int, optional
        Upscaling factor for subpixel morphological analysis (>1).
    bound_frac : float, optional
        Fraction of image mass to discard along each dimension when computing the bounding
        parallelogram.
    pool : multiprocessing.Pool, optional
        A pool of worker processes for parallel processing. Defaults to sequential computation.
    chunksize : int
        Size of the chunks in which to split the batch for parallel processing. Ignored if
        `pool=None`.

    Returns
    -------
    pandas.DataFrame
        A data frame with one row for each image, containing the following columns:

        - `area`: Total area/image mass.
        - `length`: Length of the estimated skeleton.
        - `thickness`: Mean thickness along the skeleton.
        - `slant`: Horizontal shear, in radians.
        - `width`: Width of the bounding parallelogram.
        - `height`: Height of the bounding parallelogram.

    Notes
    -----
    If the `tqdm` package is installed, this function will display a fancy progress bar with ETA.
    Otherwise, it will print a plain text progress message.
    """
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
            print(f"\rProcessing images: {0}/{len(images)}", end='')
            for i, res in enumerate(g):
                print(f"\rProcessing images: {i + 1}/{len(images)}", end='')
                yield res
            print()
        gen = plain_progress(gen)

    results = list(gen)
    columns = ['area', 'length', 'thickness', 'slant', 'width', 'height']
    df = pd.DataFrame(results, columns=columns)
    return df
