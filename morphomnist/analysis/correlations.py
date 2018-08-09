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


def partial_correlation(x, y, control, use_statsmodels=False):
    """Sample partial correlation between two variables, controlling for confounding variables.

    Parameters
    ----------
    x, y : (N,) array_like
        Variables whose partial correlation to compute.
    control : (N, M) array_like
        Confounding variables to control for.
    use_statsmodels : bool, optional
        If True, correlate the residuals from regressing `control` onto `x` and `y` using the
        `statsmodels` library, otherwise use the precision matrix explicitly, with identical
        results.

    Returns
    -------
    float
        The partial correlation, between -1 and 1.
    """
    if use_statsmodels:
        import statsmodels.api as sm
        res_x = sm.OLS(x, control).fit().resid
        res_y = sm.OLS(y, control).fit().resid
        return correlation(res_x, res_y)
    else:
        var = np.column_stack([y, x, control])
        prec = np.linalg.inv(var.T @ var)
        return -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])


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
    if which is None:
        which = np.arange(ivs.shape[1])
    else:
        which = np.atleast_1d(which)
    nx, ny = len(which), dvs.shape[1]
    pcorr = np.zeros([ny, nx])
    for j, w in enumerate(which):
        iv = ivs[:, w]
        control = np.c_[ivs[:, :w], ivs[:, w + 1:]]
        for i in range(ny):
            dv = dvs[:, i]
            pcorr[i, j] = partial_correlation(iv, dv, control)
    return pcorr


if __name__ == '__main__':
    N = 1000
    x = np.random.randn(N, 5)
    y = np.random.randn(N, 3)
    print(partial_correlation_matrix(x, y))
    print(partial_correlation_matrix(x, y, [1, 2]))
    print(partial_correlation_matrix(x, y, [1]))
    print(partial_correlation_matrix(x, y, 1))

    y_ = y[:, 0]
    print(partial_correlation_matrix(x, y_))
    print(partial_correlation_matrix(x, y_, [1, 2]))
    print(partial_correlation_matrix(x, y_, 1))
