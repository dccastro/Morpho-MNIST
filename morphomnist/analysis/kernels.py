from typing import Any, Callable, Optional, Union

import numpy as np

KernelFunc = Callable[[np.ndarray, np.ndarray, Optional[Any]], np.ndarray]


def sq_dist(x, y, scale=1.):
    """Scaled squared Euclidean distance between two collections of points.

    Parameters
    ----------
    x, y : (..., D) array_like
        Input data; must have broadcastable shapes, with data dimensions along the last axis.
    scale : optional
        - scalar: scale distances by this factor
        - (D,) array_like: scale each dimension by the corresponding value
        - (D, D) array_like: will be treated as a *squared* scale matrix (e.g. covariance)

    Returns
    -------
    np.ndarray
        The computed squared distances, with the broadcast shape of `x` and `y` minus the last axis.
    """
    x, y = np.asarray(x), np.asarray(y)
    diff = x - y
    scale = np.asarray(scale)
    if scale.ndim == 0:
        return (diff ** 2).sum(axis=-1) / (scale ** 2)
    elif scale.ndim == 1:
        return ((diff / scale) ** 2).sum(axis=-1)
    else:
        inv_scale = np.linalg.inv(scale)
        return np.einsum('...i,ij,...j->...', diff, inv_scale, diff)


def exp_quad(x: np.ndarray, y: np.ndarray, scale=1.):
    """Exponentiated quadratic (Gaussian) kernel.

    See Also
    --------
    sq_dist : Scaled squared Euclidean distance.
    """
    return np.exp(-.5 * sq_dist(x, y, scale))


def rat_quad(x: np.ndarray, y: np.ndarray, exponent: float, scale=1.):
    """Rational quadratic kernel.

    See Also
    --------
    sq_dist : Scaled squared Euclidean distance.
    """
    return (1. + .5 * sq_dist(x, y, scale) / exponent) ** (-exponent)


def _get_scale(x: np.ndarray, type: str) -> Union[float, np.ndarray]:
    if type == 'median':
        return np.sqrt(np.median(sq_dist(x[:-1, None], x[None, 1:]).flat, overwrite_input=True))
    elif type == 'rms':
        return np.sqrt(2. * np.var(x, axis=0, ddof=1).sum())
    elif type == 'std':
        return np.std(x, axis=0, ddof=1)
    elif type == 'cov':
        return np.cov(x, rowvar=False, ddof=1)
    else:
        raise ValueError("Bandwidth type must be one of ['median', 'rms', 'std', 'cov']")


def _get_factor(n: int, d: int, factor: Union[float, str]) -> float:
    if factor == 'scott':
        return n ** (-1. / (d + 4.))
    elif factor == 'silverman':
        return (n * (d + 2.) / 4.) ** (-1. / (d + 4.))
    elif isinstance(factor, str) or not np.isscalar(factor):
        raise ValueError("Factor type must be scalar, 'scott', or 'silverman'")
    return factor


def bandwidth(*x: np.ndarray, type='cov', factor='scott', subsample=None):
    """Estimates a bandwidth for kernel density estimation of the given data.

    Parameters
    ----------
    x : (N, D) array_like
        Input data, with `N` observations and `D` dimensions. If more than one array is given,
        the bandwidths are computed separately and combined such that a Gaussian kernel with this
        bandwidth is equivalent to the convolution of the individual Gaussian kernels.

    type : {'median', 'std', 'cov'}, optional
        Type of scale from which to compute the bandwidth:

        - 'median': median distance between all pairs of points (slow for large arrays)
        - 'rms': root-mean-squared distance between all pairs of points
        - 'std': standard deviations for each dimension
        - 'cov': full covariance matrix (note that this is a *squared* scale)

    factor : float or {'scott', 'silverman'}, optional
        Multiplicative factor for the scale. If not a scalar, it is computed as follows:

        - 'scott': `N**(-1/(D+4))`
        - 'silverman': `(N*(D+2)/4)**(-1/(D+4))`

    subsample : float, optional
        If not None, randomly subsample the input data without replacement by a factor of
        `subsample` (between 0 and 1). Especially useful for 'median' scaling.

    Returns
    -------
    float or np.ndarray
        The computed bandwidth ('median'/'rms': `float`, 'std': `(D,) np.ndarray`, 'cov': `(D, D)
        np.ndarray`).
    """
    if len(x) > 1:
        bws = np.array([bandwidth(ary, type=type, factor=factor, subsample=subsample) for ary in x])
        if type == 'cov':
            return bws.sum(0)
        return np.sqrt((bws ** 2).sum(0))

    x = np.asarray(x[0])
    n, d = x.shape
    if subsample is not None:
        if not (0. < subsample < 1.):
            raise ValueError(f"Subsampling factor must be in (0, 1): {subsample}")
        x = x[np.random.choice(n, size=int(subsample * n), replace=False)]

    scale = _get_scale(x, type)
    factor = _get_factor(n, d, factor)
    if type == 'cov':
        return scale * (factor ** 2)
    return scale * factor


if __name__ == '__main__':
    Nx, Ny, D = 100, 20, 5
    x = np.random.randn(Nx, D)
    y = np.random.randn(Ny, D)
    s0 = np.std(x)
    s1 = np.std(x, axis=0)
    s2 = np.cov(x, rowvar=False, bias=True)

    for s in [s0, s1, s2]:
        try:
            sq_dist(x, y, s)
            assert False
        except ValueError:
            pass
        assert sq_dist(x, x, s).shape == (Nx,)
        assert sq_dist(y, y, s).shape == (Ny,)
        assert sq_dist(x[:, None], y[None, :], s).shape == (Nx, Ny)
        assert sq_dist(y[:, None], x[None, :], s).shape == (Ny, Nx)

    print(bandwidth(x, y, type='std'))
    print(bandwidth(x, y, type='cov'))
    print(bandwidth(x, type='median'))
    print(bandwidth(x, type='rms'))
    print(bandwidth(x, type='std'))
