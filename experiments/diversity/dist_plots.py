import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker


def plot_distribution(data, cols, lims, multiples=(), formats=()):
    lims = [(l - .05 * (h - l), h + .05 * (h - l)) for l, h in lims]
    g = sns.PairGrid(data, vars=cols, diag_sharey=False, size=1.2)
    g.map_diag(plt.hist, bins=20, alpha=.4, density=True)
    g.map_diag(sns.kdeplot, legend=False)
    g.map_lower(sns.kdeplot, cmap='Blues_d', linewidths=1.)
    for i in range(g.axes.shape[0]):
        for j in range(i + 1, g.axes.shape[1]):
            g.axes[i, j].hexbin(data[cols[j]], data[cols[i]],
                                gridsize=30, bins='log', mincnt=1,
                                cmap='Blues', lw=.1, edgecolors='face',
                                extent=[*lims[j], *lims[i]], rasterized=True)

    for i, col in enumerate(cols):
        g.axes[ i, 0].set_ylim(lims[i])
        g.axes[-1, i].set_xlim(lims[i])
        if col in multiples:
            g.axes[ i, 0].yaxis.set_major_locator(ticker.MultipleLocator(multiples[col]))
            g.axes[-1, i].xaxis.set_major_locator(ticker.MultipleLocator(multiples[col]))
        if col in formats:
            g.axes[ i, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter(formats[col]))
            g.axes[-1, i].xaxis.set_major_formatter(ticker.FormatStrFormatter(formats[col]))
