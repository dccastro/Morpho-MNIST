import multiprocessing

import numpy as np
import pandas as pd
from scipy.ndimage import filters
from skimage import morphology, transform

_SKEL_LEN_MASK = np.array([[0., 0., 0.], [0., 0., 1.], [np.sqrt(2.), 1., np.sqrt(2.)]])


def _process_img_morph(img, threshold=128, scale=1):
    if scale > 1:
        up_img = transform.pyramid_expand(img, upscale=scale, order=3)  # type: np.ndarray
        img = (255. * up_img).astype(img.dtype)
    bin_img = (img >= threshold)
    skel, dist_map = morphology.medial_axis(bin_img, return_distance=True)
    return img, bin_img, skel, dist_map


class ImageMorphology(object):
    def __init__(self, image: np.ndarray, threshold: int = 128, scale: int = 1):
        self.image = image
        self.threshold = threshold
        self.scale = scale
        self.hires_image, self.binary_image, self.skeleton, self.distance_map = \
            _process_img_morph(self.image, self.threshold, self.scale)

    @property
    def area(self):
        return self.binary_image.sum() / self.scale ** 2

    @property
    def stroke_length(self):
        skel = self.skeleton.astype(float)
        conv = filters.correlate(skel, _SKEL_LEN_MASK, mode='constant')
        return np.einsum('ij,ij->', conv, skel) / self.scale

    @property
    def mean_thickness(self):
        return 2. * np.mean(self.distance_map[self.skeleton]) / self.scale

    @property
    def median_thickness(self):
        return 2. * np.median(self.distance_map[self.skeleton]) / self.scale

    def downscale(self, image):
        down_img = transform.pyramid_reduce(image, downscale=self.scale, order=3)  # type: np.ndarray
        return (255. * down_img).astype(np.uint8)


class ImageMoments(object):
    def __init__(self, img: np.ndarray):
        img = img.astype(float)
        x = np.arange(img.shape[1])[None, :]
        y = np.arange(img.shape[0])[:, None]
        m00 = img.sum()
        m10 = (x * img).sum() / m00
        m01 = (y * img).sum() / m00
        m20 = (x ** 2 * img).sum() / m00
        m11 = (x * y * img).sum() / m00
        m02 = (y ** 2 * img).sum() / m00
        self.m00 = m00
        self.m10 = m10
        self.m01 = m01
        self.u20 = m20 - m10 ** 2
        self.u11 = m11 - m10 * m01
        self.u02 = m02 - m01 ** 2

    @property
    def centroid(self):
        return self.m10, self.m01

    @property
    def covariance(self):
        return self.u20, self.u11, self.u02

    @property
    def axis_lengths(self):
        delta = .5 * np.hypot(2. * self.u11, self.u20 - self.u02)
        eig1 = .5 * (self.u20 + self.u02) + delta
        eig2 = .5 * (self.u20 + self.u02) - delta
        return np.sqrt(eig1), np.sqrt(eig2)

    @property
    def angle(self):
        return .5 * np.arctan2(2. * self.u11, self.u20 - self.u02)

    @property
    def horizontal_shear(self):
        return self.u11 / self.u02

    @property
    def vertical_shear(self):
        return self.u11 / self.u20


def _horz_cdf(img: np.ndarray, shear: float, x: np.ndarray, y: np.ndarray, y_mid):
    locs = np.arange(0, img.shape[1], step=1)
    counts = np.zeros(len(locs))
    for i, t in enumerate(locs):
        counts[i] = ((x + .5 < t + shear * (y - y_mid)) * img).sum()
    return locs, counts / img.sum()


def _vert_cdf(img: np.ndarray, y: np.ndarray):
    counts = np.zeros(img.shape[0])
    for t in range(img.shape[0]):
        counts[t] = ((y < t) * img).sum()
    return counts / img.sum()


def bounding_parallelogram(img: np.ndarray, frac: float, moments: ImageMoments = None):
    height, width = img.shape
    img = img.astype(float)
    x = np.arange(width)[None, :]
    y = np.arange(height)[:, None]

    if moments is None:
        moments = ImageMoments(img)
    middle = moments.centroid[1]
    shear = moments.horizontal_shear

    hloc, hcdf = _horz_cdf(img, shear, x, y, middle)
    vcdf = _vert_cdf(img, y)

    frac /= 2  # two-sided
    left, right = np.interp([frac, 1. - frac], hcdf, hloc)
    top, bottom = np.interp([frac, 1. - frac], vcdf, np.arange(len(vcdf)))

    top_left = np.array([left + shear * (top - middle), top])
    top_right = np.array([right + shear * (top - middle), top])
    bottom_left = np.array([left + shear * (bottom - middle), bottom])
    bottom_right = np.array([right + shear * (bottom - middle), bottom])

    return top_left, top_right, bottom_right, bottom_left


def measure_image(img: np.ndarray, threshold: int = 128, scale: int = 4, bound_frac: float = .02,
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
        print("Thickness: {:.2f}".format(mean_thck))
        print("Length: {:.1f}".format(length))
        print("Slant: {:.0f}°".format(np.rad2deg(slant)))
        print("Dimensions: {:.1f} x {:.1f}".format(width, height))
        print()

    return area, length, mean_thck, slant, width, height


def measure_batch(images: np.ndarray, threshold: int = 128, scale: int = 4, bound_frac: float = .02,
                  verbose=False, pool: multiprocessing.Pool = None, chunksize: int = 1000):
    args = ((img, threshold, scale, bound_frac, verbose) for img in images)
    if pool is None:
        results = [measure_image(*arg) for arg in args]
    else:
        results = pool.starmap(measure_image, args, chunksize=chunksize)
    columns = ['area', 'length', 'thickness', 'slant', 'width', 'height']
    df = pd.DataFrame(results, columns=columns)
    return df


if __name__ == '__main__':
    import util

    data = util.load("data/mnist/patho/t10k-images-idx3-ubyte.gz")[:100]
    with multiprocessing.Pool() as pool:
        df = measure_batch(data, pool=pool, chunksize=10)
    print(df.head())
    print(measure_batch(data).head())
