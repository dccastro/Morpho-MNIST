import numpy as np
from skimage import draw, morphology, transform

import skeleton
from morpho import ImageMorphology


def _sample_coords(skel):
    coords = np.array(np.where(skel)).T
    centre_idx = np.random.choice(coords.shape[0])
    return coords[centre_idx]


class Operator(object):
    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        raise NotImplementedError


class ThinOperator(Operator):
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, morph: ImageMorphology):
        radius = int(self.amount * morph.scale)
        return morphology.erosion(morph.binary_image, morphology.disk(radius))


class ThickenOperator(Operator):
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, morph: ImageMorphology):
        radius = int(self.amount * morph.scale)
        return morphology.dilation(morph.binary_image, morphology.disk(radius))


class DeformationOperator(Operator):
    def __call__(self, morph: ImageMorphology):
        return transform.warp(morph.binary_image, lambda xy: self.warp(xy, morph))

    def warp(self, xy: np.ndarray, morph: ImageMorphology):
        raise NotImplementedError


class SwellOperator(DeformationOperator):
    def __init__(self, strength: float, radius: float):
        self.strength = strength
        self.radius = radius

    def warp(self, xy: np.ndarray, morph: ImageMorphology):
        centre = _sample_coords(morph.skeleton)[::-1]
        radius = self.radius * morph.scale

        offset_xy = xy - centre
        distance = np.hypot(*offset_xy.T)
        weight = (distance / radius) ** (self.strength - 1)
        weight[distance > radius] = 1.
        return centre + weight[:, None] * offset_xy


class FractureOperator(Operator):
    _ANGLE_WINDOW = 2
    _FRAC_EXTENSION = .5

    def __init__(self, thickness: float = 1.5, prune: float = 2, num_frac: int = 1):
        self.thickness = thickness
        self.prune = prune
        self.num_frac = num_frac

    def __call__(self, morph: ImageMorphology):
        skel = morph.skeleton
        up_prune = self.prune * morph.scale
        pruned = skeleton.erase(skel, skeleton.num_neighbours(skel) == 1, up_prune)
        forked = skeleton.erase(pruned, skeleton.num_neighbours(pruned) == 3, up_prune)

        up_thickness = self.thickness * morph.scale
        r = int(np.ceil((up_thickness - 1) / 2))
        brush = ~morphology.disk(r).astype(bool)
        frac_img = np.pad(morph.binary_image, pad_width=r, mode='constant', constant_values=False)
        for _ in range(self.num_frac):
            centre = _sample_coords(forked)
            p0, p1 = self._endpoints(skel, morph, centre)
            self._draw_line(frac_img, p0, p1, brush)
        return frac_img[r:-r, r:-r]

    def _endpoints(self, skel, morph, centre):
        angle = skeleton.get_angle(skel, *centre, self._ANGLE_WINDOW * morph.scale)
        length = morph.distance_map[centre[0], centre[1]] + self._FRAC_EXTENSION * morph.scale
        normal = length * np.array([np.cos(angle), -np.sin(angle)])
        p0 = (centre + normal).astype(int)
        p1 = (centre - normal).astype(int)
        return p0, p1

    @staticmethod
    def _draw_line(img, p0, p1, brush):
        h, w = brush.shape
        ii, jj = draw.line(*p0, *p1)
        for i, j in zip(ii, jj):
            img[i:i + h, j:j + w] &= brush


def op_thin(img, strength):
    return morphology.erosion(img, morphology.disk(strength))


def op_thicken(img, strength):
    return morphology.dilation(img, morphology.disk(strength))


def op_swell(img, strength, centre, radius):
    def inv_warp(xy):
        offset_xy = xy - centre
        distance = np.hypot(*offset_xy.T)
        # weight = 1. - strength * np.exp(-.5 * distance ** 2 / radius ** 2)
        # weight = np.minimum(1. - strength * (1. - (distance / radius) ** 2), 1.)
        weight = (distance / radius) ** (strength - 1)
        weight[distance > radius] = 1.
        return centre + weight[:, None] * offset_xy
        # return xy + strength * np.sqrt(radius) * offset_xy

    return transform.warp(img, inv_warp)


def op_power(img, strength, centre, radius):
    return transform.warp(img, lambda xy: _power_bkd(xy, strength, centre, radius))


def _radial(xy, centre, radius, radial_fn):
    offset_xy = xy - centre
    distance = np.hypot(*offset_xy.T) + np.finfo(float).eps
    weight = radial_fn(distance / radius)
    weight[distance > radius] = 1.
    return centre + weight[:, None] * offset_xy


def _power_bkd(xy, strength, centre, radius):
    offset_xy = xy - centre
    distance = np.hypot(*offset_xy.T) + np.finfo(float).eps
    weight = (distance / radius) ** (strength - 1)
    weight[distance > radius] = 1.
    return centre + weight[:, None] * offset_xy


def _power_fwd(xy_, strength, centre, radius):
    return _power_bkd(xy_, 1. / strength, centre, radius)


if __name__ == '__main__':
    x = np.arange(40)
    y = np.arange(40)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    # xy_ = _sphere_bkd(xy, .999999, (20, 20), 15)
    # xy_ = _radial(xy, (20, 20), 15, lambda r: _sphere_radial_fwd(r, .95))
    xy_ = _power_bkd(xy, 2, (20, 20), 12)
    # xy_ = _power_bkd(xy_, 1.5, (20, 20), 12)
    print(xy.shape)
    print(xy_.shape)
    import matplotlib.pyplot as plt

    # plt.plot(*xy_.T, '.')
    plt.pcolormesh(xy_[:, 0].reshape(xx.shape), xy_[:, 1].reshape(xx.shape), np.zeros_like(xx),
                   facecolor='none',
                   edgecolor='k',
                   cmap='gray', clim=[0, 1],
                   antialiased=True, lw=.5)
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.show()
