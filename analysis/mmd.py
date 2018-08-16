from typing import Tuple

import numpy as np

from . import kernels


def _h_moments(x: np.ndarray, y: np.ndarray, I, J, kernel: kernels.KernelFunc,
               skip_diag: bool = False, **kwargs) -> np.ndarray:
    H = + kernel(x[I], x[J], **kwargs) + kernel(y[I], y[J], **kwargs) \
        - kernel(x[I], y[J], **kwargs) - kernel(y[I], x[J], **kwargs)
    sums = np.array([H.size, H.sum(), (H ** 2).sum()])
    # dH = H[I == J]  # Slower but more correct version without assumptions on I and J
    if skip_diag:
        dH = np.diag(H)
        sums -= dH.size, dH.sum(), (dH ** 2).sum()
    return sums


def mmd2(x: np.ndarray, y: np.ndarray, kernel: kernels.KernelFunc = kernels.exp_quad,
         linear: bool = True, unbiased: bool = True, chunksize: int = None, seed: int = None,
         **kwargs) -> Tuple[float, float]:
    """Squared maximum mean discrepancy (MMD) between two sets of observations [1]_.

    Parameters
    ----------
    x, y : (N, D) array_like
        Input samples whose empirical MMD to estimate.
    kernel : callable, optional
        A kernel function, taking arrays as first two arguments (e.g. see `kernels` module).
    linear : bool, optional
        If True, computes the linear-time statistic.
    unbiased : bool, optional
        If True, computes the unbiased statistic, otherwise returns the biased statistic.
        This option is ignored if `linear=True`.
    chunksize : int, optional
        If not None, splits `x` and `y` in pieces of size `chunksize` to compute and aggregate
        intermediate results. Useful when `N` is large, as `O(N^2)` storage would be required.
    seed : int, optional
        Seed for the pseudo-random number generator used in shuffling the input samples. If
        none, uses the default PRNG. Useful to ensure data is shuffled consistently between calls.
    **kwargs
        Additional keyword arguments for the kernel function: `kernel(x, y, **kwargs)`.

    Returns
    -------
    (float, float)
        The estimate and standard error of the squared MMD. The standard error is NaN when
        computing the biased statistic.

    References
    ----------
    .. [1] Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. J. (2012). A
       Kernel Two-Sample Test. Journal of Machine Learning Research, 13(Mar), 723–773.
    """
    x, y = np.asarray(x), np.asarray(y)
    m = min(x.shape[0], y.shape[0])
    prng = np.random if seed is None else np.random.RandomState(seed)
    x = prng.permutation(x)[:m]
    y = prng.permutation(y)[:m]
    size = m // 2 if linear else m
    if chunksize is None:
        chunksize = size
    get_chunk = lambda k: np.arange(k * chunksize, min((k + 1) * chunksize, size))
    moments = np.zeros(3)
    for i in range(0, size // chunksize + 1):
        I = get_chunk(i)[:, None]
        if linear:
            moments += _h_moments(x, y, 2 * I, 2 * I + 1, kernel, **kwargs)
        else:
            for j in range(0, size // chunksize + 1):
                J = get_chunk(j)[None, :]
                skip_diag = unbiased and (i == j)
                moments += _h_moments(x, y, I, J, kernel, skip_diag=skip_diag, **kwargs)
    mean = moments[1] / moments[0]
    # sem = np.sqrt((moments[2] / moments[0] - mean ** 2) / moments[0])
    var = np.nan
    if linear:
        var = (moments[2] / moments[0] - mean ** 2) / moments[0]
    elif unbiased:
        var = 2. * (moments[2] / moments[0] - mean ** 2) / moments[0]
    sem = np.sqrt(var)
    return mean, sem


def bound_threshold_biased(alpha, m, K=1.):
    return (2. * K / m) * (1. + np.sqrt(-2. * np.log(alpha))) ** 2


def bound_threshold_unbiased(alpha, m, K=1.):
    return (4. * K / np.sqrt(m)) * np.sqrt(-np.log(alpha))


def asymptotic_linear_test(mean: float, sem: float) -> Tuple[float, float]:
    """Asymptotic two-sample test based on the linear-time squared MMD statistic [1]_.

    Parameters
    ----------
    mean : float
        Linear-time estimate of the squared MMD.
    sem : float
        Standard error of the linear-time estimate of the squared MMD.

    Returns
    -------
    (float, float)
        The normalised test statistic and p-value.

    References
    ----------
    .. [1] Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. J. (2012). A
       Kernel Two-Sample Test. Journal of Machine Learning Research, 13(Mar), 723–773.
    """
    from scipy import stats
    z = mean / sem
    pval = stats.norm.sf(z)
    return z, pval


def test(x: np.ndarray, y: np.ndarray, kernel: kernels.KernelFunc = kernels.exp_quad,
         linear: bool = False, unbiased: bool = True, chunksize: int = None, seed: int = None,
         **kwargs) -> Tuple[float, float, float]:
    """Kernel two-sample test.

    This is a convenience wrapper function to compute the MMD estimate and perform the asymptotic
    linear test (if `linear=True`), with human-readable output. It is equivalent to calling `mmd2()`
    followed by `asymptotic_linear_test()`. See their documentation for details on parameters and
    return values.

    Returns
    -------
    (float, float, float)
        The estimated squared MMD, standard error and asymptotic test p-value

    See Also
    --------
    mmd2, asymptotic_linear_test
    """
    mean, sem = mmd2(x, y, kernel, linear, unbiased, chunksize, seed, **kwargs)
    pval = np.nan
    text = f"MMD\u00b2_{'l' if linear else ('u'  if unbiased else 'b')} \u2248 {mean:.6g}"
    if np.isfinite(sem):
        text += f" \u00b1 {sem:.6g}"
    if linear:
        z, pval = asymptotic_linear_test(mean, sem)
        text += f" (z={z:.4g}, p={pval:.4f})"
    print(text)
    return mean, sem, pval


if __name__ == '__main__':
    n, m, d = 1000, 1000, 5
    x = np.random.randn(n, d)
    y = np.random.randn(m, d)
    z = np.random.randn(m, d)
    y[:, 0] += 1.

    seed = np.random.randint(10000)
    bw_kwargs = dict(type='cov', factor='scott')
    for chunksize in [200]:
        scale = kernels.bandwidth(x, **bw_kwargs) + kernels.bandwidth(y, **bw_kwargs)
        print(np.diag(scale))
        test(x, y, linear=True, chunksize=chunksize, seed=seed, scale=scale)
        test(x, y, unbiased=True, chunksize=chunksize, seed=seed, scale=scale)
        test(x, y, unbiased=False, chunksize=chunksize, seed=seed, scale=scale)
        print()

        scale = kernels.bandwidth(x, **bw_kwargs) + kernels.bandwidth(z, **bw_kwargs)
        print(np.diag(scale))
        test(x, z, linear=True, chunksize=chunksize, seed=seed, scale=scale)
        test(x, z, unbiased=True, chunksize=chunksize, seed=seed, scale=scale)
        test(x, z, unbiased=False, chunksize=chunksize, seed=seed, scale=scale)
        print()
    for scale in np.arange(.5, 5, .5)[:0]:
        print(mmd2(x, y, unbiased=True, chunksize=200, seed=seed, scale=scale))
        print(mmd2(x, z, unbiased=True, chunksize=200, seed=seed, scale=scale))
        print()
