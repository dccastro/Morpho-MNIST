import numpy as np
from scipy.ndimage import filters
from skimage import morphology, transform

_SKEL_LEN_MASK = np.array([[0., 0., 0.], [0., 0., 1.], [np.sqrt(2.), 1., np.sqrt(2.)]])


def _process_img_morph(img, threshold=128, upscale=1):
    if upscale > 1:
        up_img = transform.pyramid_expand(img, upscale=upscale, order=3)  # type: np.ndarray
        img = (255. * up_img).astype(img.dtype)
    bin_img = (img >= threshold)
    skel, dist_map = morphology.medial_axis(bin_img, return_distance=True)
    return img, bin_img, skel, dist_map


class ImageMorphology(object):
    def __init__(self, image: np.ndarray, threshold: int = 128, upscale: int = 1):
        self.image = image
        self.threshold = threshold
        self.upscale = upscale
        self.hires_image, self.binary_image, self.skeleton, self.distance_map = \
            _process_img_morph(self.image, self.threshold, self.upscale)

    @property
    def area(self):
        return self.binary_image.sum() / self.upscale ** 2

    @property
    def stroke_length(self):
        skel = self.skeleton.astype(float)
        conv = filters.correlate(skel, _SKEL_LEN_MASK, mode='constant')
        return np.einsum('ij,ij->', conv, skel) / self.upscale

    @property
    def mean_thickness(self):
        return 2. * np.mean(self.distance_map[self.skeleton]) / self.upscale

    @property
    def median_thickness(self):
        return 2. * np.median(self.distance_map[self.skeleton]) / self.upscale


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
