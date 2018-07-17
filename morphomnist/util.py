import gzip

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from . import idx


def plot_digit(x, ax=None, title=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    def_kwargs = dict(cmap='gray_r')
    def_kwargs.update(**kwargs)
    ax.imshow(x.squeeze(), **def_kwargs)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if title is not None:
        ax.set_title(title)


def plot_grid(imgs, nrow=None, **kwargs):
    num = imgs.shape[0]
    if nrow is None:
        nrow = int(np.floor(np.sqrt(num)))
    ncol = int(np.ceil(num / nrow))
    fig, axs = plt.subplots(nrow, ncol, **kwargs)
    axs = np.atleast_1d(axs).T
    for i in range(num):
        ax = axs.flat[i]
        plot_digit(imgs[i], ax)
        ax.axis('off')
    for ax in axs.flat[num:]:
        ax.set_visible(False)
    return fig, axs


def plot_ellipse(x, y, angle, major, minor, ax, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.add_patch(Ellipse(xy=(x, y), width=2 * major, height=2 * minor,
                         angle=np.rad2deg(angle), **kwargs))


def plot_parallelogram(top_left, top_right, bottom_right, bottom_left, scale=1., ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    corners = [top_left, top_right, bottom_right, bottom_left, top_left]
    ax.plot(*(scale * np.array(corners).T - .5), **kwargs)


def save(data, path):
    with gzip.open(path, 'wb') as f:
        idx.save_uint8(data, f)


def load(path):
    with gzip.open(path, 'rb') as f:
        data = idx.load_uint8(f)
    return data
