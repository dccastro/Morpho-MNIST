import numpy as np
from skimage import morphology, transform

from morpho import ImageMorphology


class Operator(object):
    def __call__(self, morpho: ImageMorphology) -> np.ndarray:
        raise NotImplementedError


class ThinOperator(Operator):
    def __init__(self, amount):
        self.amount = amount
        self._selem = morphology.disk(self.amount)

    def __call__(self, morpho: ImageMorphology):
        return morphology.erosion(morpho.binary_image, self._selem)


class ThickenOperator(Operator):
    def __init__(self, amount):
        self.amount = amount
        self._selem = morphology.disk(self.amount)

    def __call__(self, morpho: ImageMorphology):
        return morphology.dilation(morpho.binary_image, self._selem)


class DeformationOperator(Operator):
    def __init__(self, warp_fn, *args, **kwargs):
        self.warp_fn = warp_fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, morpho: ImageMorphology):
        return transform.warp(morpho.binary_image,
                              lambda xy: self.warp_fn(xy, *self.args, **self.kwargs))


class RandomLocationOperator(Operator):
    def __init__(self, def_op: DeformationOperator):
        self.def_op = def_op

    def __call__(self, morpho: ImageMorphology):
        skel_idx = np.where(morpho.skeleton)
        centre_idx = np.random.choice(len(skel_idx[0]))
        centre = (skel_idx[1][centre_idx], skel_idx[0][centre_idx])
        self.def_op.kwargs['centre'] = centre
        return self.def_op(morpho)


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


def op_sphere(img, strength, centre, radius):
    return transform.warp(img, lambda xy: _sphere_fwd(xy, strength, centre, radius))


def _sphere_bkd(xy, strength, c, R0):
    if strength >= 1.:
        raise ValueError("strength must be < 1")
    c = np.array(c)
    xy = xy - c
    r = np.hypot(*xy.T) + np.finfo(float).eps
    gamma = strength * np.pi / 2
    R = R0 / np.sin(gamma)
    alpha = np.arcsin(r / R)
    r_ = alpha / gamma * R0 / r
    xy_ = r_[:, None] * xy
    xy_[r > R0] = xy[r > R0]
    return c + xy_


def _sphere_radial_bkd(r, strength):
    if strength >= 1.:
        raise ValueError("strength must be < 1")
    gamma = strength * np.pi / 2
    R = 1. / np.sin(gamma)
    alpha = np.arcsin(r / R)
    r_ = alpha / (r * gamma)
    r_[r > 1.] = 1.
    return r_


def _sphere_radial_fwd(r_, strength):
    if strength >= 1.:
        raise ValueError("strength must be < 1")
    gamma = strength * np.pi / 2
    R = 1. / np.sin(gamma)
    alpha = r_ * gamma
    r = np.sin(alpha) * R / r_
    r[r_ > 1.] = 1.
    return r


def _sphere_fwd(xy_, strength, c, R0):
    if strength >= 1.:
        raise ValueError("strength must be < 1")
    c = np.array(c)
    xy_ = xy_ - c
    r_ = np.hypot(*xy_.T) + np.finfo(float).eps
    gamma = strength * np.pi / 2
    R = R0 / np.sin(gamma)
    alpha = r_ / R0 * gamma
    r = np.sin(alpha) * R / r_
    xy = r[:, None] * xy_
    xy[r_ > R0] = xy_[r_ > R0]
    return c + xy


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
