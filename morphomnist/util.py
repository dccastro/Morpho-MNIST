import gzip

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


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


def plot_grid(imgs, nrow=None, digit_kw=None, **kwargs):
    num = imgs.shape[0]
    if nrow is None:
        nrow = int(np.floor(np.sqrt(num)))
    ncol = int(np.ceil(num / nrow))
    fig, axs = plt.subplots(nrow, ncol, **kwargs)
    axs = np.atleast_1d(axs).T
    if digit_kw is None:
        digit_kw = {}
    for i in range(num):
        ax = axs.flat[i]
        plot_digit(imgs[i], ax, **digit_kw)
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
