from typing import Sequence

import numpy as np
from skimage import draw, morphology, transform

from . import skeleton
from .morpho import ImageMorphology


class Perturbation(object):
    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        raise NotImplementedError


class Thinning(Perturbation):
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, morph: ImageMorphology):
        radius = int(self.amount * morph.scale * morph.mean_thickness / 2.)
        return morphology.erosion(morph.binary_image, morphology.disk(radius))


class Thickening(Perturbation):
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, morph: ImageMorphology):
        radius = int(self.amount * morph.scale * morph.mean_thickness / 2.)
        return morphology.dilation(morph.binary_image, morphology.disk(radius))


class Deformation(Perturbation):
    def __call__(self, morph: ImageMorphology):
        return transform.warp(morph.binary_image, lambda xy: self.warp(xy, morph))

    def warp(self, xy: np.ndarray, morph: ImageMorphology):
        raise NotImplementedError


class Swelling(Deformation):
    def __init__(self, strength: float, radius: float):
        self.strength = strength
        self.radius = radius
        self.loc_sampler = skeleton.LocationSampler()

    def warp(self, xy: np.ndarray, morph: ImageMorphology):
        centre = self.loc_sampler.sample(morph)[::-1]
        radius = (self.radius * np.sqrt(morph.mean_thickness) / 2.) * morph.scale

        offset_xy = xy - centre
        distance = np.hypot(*offset_xy.T)
        weight = (distance / radius) ** (self.strength - 1)
        weight[distance > radius] = 1.
        return centre + weight[:, None] * offset_xy


class Fracture(Perturbation):
    _ANGLE_WINDOW = 2
    _FRAC_EXTENSION = .5

    def __init__(self, thickness: float = 1.5, prune: float = 2, num_frac: int = 1):
        self.thickness = thickness
        self.prune = prune
        self.num_frac = num_frac
        self.loc_sampler = skeleton.LocationSampler(prune, prune)

    def __call__(self, morph: ImageMorphology):
        up_thickness = self.thickness * morph.scale
        r = int(np.ceil((up_thickness - 1) / 2))
        brush = ~morphology.disk(r).astype(bool)
        frac_img = np.pad(morph.binary_image, pad_width=r, mode='constant', constant_values=False)
        for centre in self.loc_sampler.sample(morph, self.num_frac):
            p0, p1 = self._endpoints(morph, centre)
            self._draw_line(frac_img, p0, p1, brush)
        return frac_img[r:-r, r:-r]

    def _endpoints(self, morph, centre):
        angle = skeleton.get_angle(morph.skeleton, *centre, self._ANGLE_WINDOW * morph.scale)
        length = morph.distance_map[centre[0], centre[1]] + self._FRAC_EXTENSION * morph.scale
        angle += np.pi / 2.  # Perpendicular to the skeleton
        normal = length * np.array([np.sin(angle), np.cos(angle)])
        p0 = (centre + normal).astype(int)
        p1 = (centre - normal).astype(int)
        return p0, p1

    @staticmethod
    def _draw_line(img, p0, p1, brush):
        h, w = brush.shape
        ii, jj = draw.line(*p0, *p1)
        for i, j in zip(ii, jj):
            img[i:i + h, j:j + w] &= brush


class RandomPerturbation(Perturbation):
    def __init__(self, ops: Sequence[Perturbation]):
        self.ops = ops

    def __call__(self, morph: ImageMorphology):
        op = np.random.choice(self.ops)
        return op(morph)
