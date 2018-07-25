import numpy as np
import statsmodels.api as sm


def correlation(x, y):
    return ((x - x.mean()) / x.std(0)).T @ ((y - y.mean(0)) / y.std(0)) / x.shape[0]


def partial_correlation(x, y, conf):
    res_x = sm.OLS(x, conf).fit().resid
    res_y = sm.OLS(y, conf).fit().resid
    return correlation(res_x, res_y)


def partial_correlation2(x, y, conf):
    var = np.column_stack([y, x, conf])
    prec = np.linalg.inv(var.T @ var)
    return -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])


def compute_infogan_pcorr(phi, mu, gamma, metrics, cols):
    cat_dim, cont_dim, bin_dim = phi.shape[1], mu.shape[1], gamma.shape[1]
    pcorr = np.zeros([len(cols), cat_dim + cont_dim + bin_dim])
    phi_ = sm.categorical(phi.argmax(1), drop=True)
    gamma_ = gamma > .5
    for c, col in enumerate(cols):
        dv = metrics[col].values
        for cat in range(cat_dim):
            iv = phi_[:, cat]
            conf = np.column_stack([
                np.ones(iv.shape[0]),
                mu,
                gamma_
            ])
            pcorr[c, cat] = partial_correlation2(iv, dv, conf)
        for cont in range(cont_dim):
            iv = mu[:, cont]
            conf = np.column_stack([
                phi_,
                mu[:, :cont],
                mu[:, cont + 1:],
                gamma_
            ])
            pcorr[c, cat_dim + cont] = partial_correlation2(iv, dv, conf)
        for bin in range(bin_dim):
            iv = gamma_[:, bin]
            conf = np.column_stack([
                phi_,
                mu,
                gamma_[:, :bin],
                gamma_[:, bin + 1:]
            ])
            pcorr[c, cat_dim + cont_dim + bin] = partial_correlation2(iv, dv, conf)
    return pcorr
