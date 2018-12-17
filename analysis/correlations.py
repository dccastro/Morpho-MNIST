import numpy as np


def correlation(x, y):
    """Sample correlation between two variables.

    Parameters
    ----------
    x, y : 1D or 2D array_like
        Variables whose correlation to compute.

    Returns
    -------
    float or np.ndarray
        The correlation between `x` and `y`, with values between -1 and 1. The shape (scalar, 1D, or
        2D) will depend on the number of columns of the inputs.
    """
    x, y = np.asarray(x), np.asarray(y)
    x_ = (x - x.mean(0)) / x.std(0)
    y_ = (y - y.mean(0)) / y.std(0)
    return x_.T @ y_ / x.shape[0]


def partial_correlations(ivs, dv):
    """Sample partial correlations between two variables, controlling for confounding variables.

    Parameters
    ----------
    dv : (N,) array_like
        Target dependent variable.
    ivs : (N, M) array_like
        Independent variables. The partial correlation will be computed for each column in turn,
        while controlling for the remaining columns.

    Returns
    -------
    (M,) np.ndarray
        The partial correlations, between -1 and 1.
    """
    ivs, dv = np.asarray(ivs), np.asarray(dv)
    if dv.ndim == 1:
        dv = dv[:, None]
    var = np.column_stack([dv, ivs])
    var -= var.mean(axis=0)
    prec = np.linalg.inv(var.T @ var)
    return -prec[0, 1:] / np.sqrt(prec[0, 0] * np.diag(prec)[1:])


def partial_correlation_matrix(ivs, dvs, which=None):
    """Sample partial correlations of each dependent variable with each selected independent
    variable, while controlling for all remaining independent variables at a time.

    Parameters
    ----------
    ivs : (N, M) array_like
        Independent variables.
    dvs : (N, I) array_like
        Dependent variables. If a 1D-array is given, it will be reshaped to `(N, 1)`.
    which : (J,) array_like, optional
        Column indices of the independent variables for which to compute the partial correlations.
        If None (default), compute for all columns of `ivs`. If an integer is given, it will be cast
        as a `(1,)` array.

    Returns
    -------
    (I, J) np.ndarray
        Partial correlation matrix between dependent variables (rows) and selected independent
        variables (columns).
    """
    ivs, dvs = np.asarray(ivs), np.asarray(dvs)
    if dvs.ndim == 1:
        dvs = dvs[:, None]
    pcorr = np.asarray([partial_correlations(ivs, dv) for dv in dvs.T])
    if which is not None:
        pcorr = pcorr[..., np.atleast_1d(which)]
    return pcorr


if __name__ == '__main__':
    N = 1000
    x = np.random.randn(N, 5) + 1.
    y = x @ np.random.randn(5, 3) + np.random.randn(N, 3)
    print(partial_correlation_matrix(x, y))
    print(partial_correlation_matrix(x, y, [1, 2]))
    print(partial_correlation_matrix(x, y, [1]))
    print(partial_correlation_matrix(x, y, 1))

    y_ = y[:, 0]
    print(partial_correlation_matrix(x, y_))
    print(partial_correlation_matrix(x, y_, [1, 2]))
    print(partial_correlation_matrix(x, y_, 1))
