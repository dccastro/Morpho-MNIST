"""Mutual Information Gap

Adapted from the implementation in Google's `disentanglement_lib`:
https://github.com/google-research/disentanglement_lib
"""
from typing import Iterable

import numpy as np
from sklearn.metrics import mutual_info_score


def _discrete_mutual_info(x, y):
    return np.array([[mutual_info_score(yj, xi) for yj in y] for xi in x])


def _discrete_entropy(x):
    return np.array([mutual_info_score(xi, xi) for xi in x])


def _is_float(x):
    return np.issubsctype(x, np.floating)


def _discretize(x, discretize, bins):
    x = np.asarray(x)
    if discretize is None:
        discretize = _is_float(x)
    if not isinstance(discretize, Iterable):
        discretize = [discretize] * x.shape[1]
    if x.shape[1] != len(discretize):
        raise ValueError(f"Expected 1 or {x.shape[1]} discretization flags, got {len(discretize)}")
    return [np.digitize(xi, np.histogram_bin_edges(xi, bins)[:-1]) if disc else xi
            for xi, disc in zip(x.T, discretize)]


def mig(codes: np.ndarray, factors: np.ndarray, discretize_codes=None, discretize_factors=None,
        bins=20):
    """Mutual information gap score (MIG) [1]_.

    Parameters
    ----------
    codes : (N, C) array_like
        Latent representations inferred by a model.
    factors : (N, F) array_like
        Generative factor annotations.
    discretize_codes : bool or sequence of bool, optional
        Whether the given `codes` should be discretized. If None (default), float inputs will be
        discretized.
    discretize_factors : bool or sequence of bool, optional
        Whether the given `factors` should be discretized. If None (default), float inputs will be
        discretized.
    bins : int or sequence of scalars or str, optional
        Argument to the discretization function (`np.histogram_bin_edges`): number of bins,
        sequence of bin edges, or name of the method to compute optimal bin width. Is ignored if
        neither `codes` nor `factors` need discretizing.

    Returns
    -------
    mig_score : float
        The computed MIG score.
    mi : (C, F) np.ndarray
        The mutual information matrix.
    entropy : (F,) np.ndarray
        The entropies for each generative factor.

    See Also
    --------
    np.histogram_bin_edges

    References
    ----------
    .. [1] Chen, T. Q., Li, X., Grosse, T. B.. & Duvenaud, D. K. (2018). Isolating Sources of
       Disentanglement in Variational Autoencoders. In Advances in Neural Information Processing
       Systems 31 (NeurIPS 2018), pp. 2610-2620.
    """
    codes = _discretize(codes, discretize_codes, bins)
    factors = _discretize(factors, discretize_factors, bins)

    mi = _discrete_mutual_info(codes, factors)
    entropy = _discrete_entropy(factors)
    sorted_mi = np.sort(mi, axis=0)[::-1]
    return np.mean((sorted_mi[0] - sorted_mi[1]) / entropy), mi, entropy


if __name__ == '__main__':
    num_samples = 10000
    dims_x, dims_y = 2, 3
    x = np.random.randn(num_samples, dims_x)
    y = np.random.randn(num_samples, dims_y)
    x[:, 0] = y[:, 0]  # + .5 * x[0]
    x[:, 1] = y[:, 1]

    bin_edges = [np.histogram_bin_edges(yj, 25) for yj in y.T]
    y_dig = [np.digitize(yj, ej[:-1]) for yj, ej in zip(y.T, bin_edges)]
    y_ = np.asarray([ej[dj] for ej, dj in zip(bin_edges, y_dig)]).T

    # for bins in [8, 16, 24, 32, 40, 48, 56, 64]:
    #     print(bins, mig(x, y, bins=bins))
    # print(mig(x, y_))
    print(mig(x, y_, discretize_factors=True))
    print(mig(x, y_, discretize_factors=False))
    print(mig(x, y_, discretize_factors=[True, False, False]))
