import numpy as np
from skimage import morphology, transform


class ImageMorphology(object):
    def __init__(self, image: np.ndarray, threshold: int = 128, upscale: int = 1):
        self.image = image
        self.threshold = threshold
        self.upscale = upscale
        self.binary_image, self.skeleton, self.distance_map = \
            preprocess_img(self.image, self.threshold, self.upscale)

    @property
    def area(self):
        return self.binary_image.sum() / self.upscale ** 2

    @property
    def stroke_length(self):
        return self.skeleton.sum() / self.upscale

    @property
    def mean_thickness(self):
        return 2. * np.mean(self.distance_map[self.skeleton]) / self.upscale

    @property
    def median_thickness(self):
        return 2. * np.median(self.distance_map[self.skeleton]) / self.upscale


def preprocess_img(img, threshold=128, upscale=1):
    if upscale > 1:
        img = (255. * transform.pyramid_expand(img, upscale=upscale)).astype(img.dtype)
    bin_img = (img >= threshold)
    skel, dist_map = morphology.medial_axis(bin_img, return_distance=True)
    return bin_img, skel, dist_map
