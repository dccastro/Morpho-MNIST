import numpy as np
from skimage import morphology, transform

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
        self._selem = morphology.disk(self.amount)

    def __call__(self, morph: ImageMorphology):
        return morphology.erosion(morph.binary_image, self._selem)


class ThickenOperator(Operator):
    def __init__(self, amount):
        self.amount = amount
        self._selem = morphology.disk(self.amount)

    def __call__(self, morph: ImageMorphology):
        return morphology.dilation(morph.binary_image, self._selem)


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



class RandomLocationOperator(Operator):
    def __init__(self, def_op: DeformationOperator):
        self.def_op = def_op

    def __call__(self, morph: ImageMorphology):
        skel_idx = np.where(morph.skeleton)
        centre_idx = np.random.choice(len(skel_idx[0]))
        centre = (skel_idx[1][centre_idx], skel_idx[0][centre_idx])
        self.def_op.kwargs['centre'] = centre
        return self.def_op(morph)


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
