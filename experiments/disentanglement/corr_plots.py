import matplotlib.pyplot as plt
import numpy as np
from seaborn.utils import relative_luminance


def annotate_corr_plot(corr, cmap=None, clim=(-1, 1), ax=None):
    ax = ax if ax is not None else plt.gca()

    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if cmap is None:
                invert = abs(corr[i, j]) > .5
            else:
                color = cmap((corr[i, j] - clim[0]) / (clim[1] - clim[0]))
                invert = relative_luminance(color) <= .408
            text_color = 'w' if invert else 'k'
            text = str(int(100. * corr[i, j]))
            ax.annotate(text, xy=(j, i), ha='center', va='center', color=text_color)


def table(corr, cmap='RdBu', vmax=1., ax=None):
    if ax is None:
        ax = plt.gca()
    clim = (-vmax, vmax)
    ax.matshow(corr, clim=clim, cmap=cmap)
    annotate_corr_plot(corr, cmap, clim, ax)


def hinton(corr, vmax=1., colors=('white', 'black'), bgcolor='gray', cmap=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    if cmap is not None:
        bgcolor = cmap(.5)
    ax.set_facecolor(bgcolor)
    ax.set_aspect('equal', 'box')

    for (y, x), w in np.ndenumerate(corr):
        if cmap is None:
            color = colors[int(w <= 0)]
        else:
            color = cmap(.5 * (w / vmax + 1.))
        size = np.sqrt(np.abs(w) / vmax)
        # patch = plt.Rectangle([x - size / 2, y - size / 2], size, size,
        #                       facecolor=color, edgecolor=color, lw=0)
        patch = plt.Circle([x, y], size / 2, facecolor=color, edgecolor=color, lw=0)
        ax.add_patch(patch)

    ax.set_xlim(-.5, corr.shape[1] - .5)
    ax.set_ylim(corr.shape[0] - .5, -.5)


def bars(corr, vmax=1., colors=('b', 'r'), bgcolor='w', ax=None):
    ax = ax if ax is not None else plt.gca()

    ax.set_facecolor(bgcolor)
    ax.set_aspect('equal', 'box')

    x = np.arange(corr.shape[1])
    y = np.arange(corr.shape[0])
    idx = np.empty((x.shape[0], y.shape[0], 2), int)
    idx[:, :, 0] = x[:, None]
    idx[:, :, 1] = y[None, :]
    idx = idx.reshape(-1, 2)

    corr_ = corr.T.flatten()

    ax.bar(idx[:, 0], -.5 * corr_ / vmax, .5, idx[:, 1],
           color=np.array(colors)[(corr_ <= 0).astype(int)])
    ax.set_xlim(-.5, corr.shape[1] - .5)
    ax.set_ylim(corr.shape[0] - .5, -.5)
