from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def plot_digit(img, ax=None, title=None, **kwargs):
    """Plots a single MNIST digit.

    Parameters
    ----------
    img : (H, W) array_like
        2D array containing the digit image.
    ax : matplotlib.axes.Axes, optional
        Axes onto which to plot. Defaults to current axes.
    title : str, optional
        If given, sets the plot's title.
    **kwargs
        Keyword arguments passed to `plt.imshow(...)`.
    """
    if ax is None:
        ax = plt.gca()
    def_kwargs = dict(cmap='gray_r')
    def_kwargs.update(**kwargs)
    ax.imshow(np.asarray(img).squeeze(), **def_kwargs)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if title is not None:
        ax.set_title(title)


def plot_grid(imgs, nrow=None, digit_kw=None, **kwargs):
    """Plots a grid of MNIST digits.

    Parameters
    ----------
    imgs : (N, H, W) array_like
        3D array containing the digit images, indexed along the first axis.
    nrow : int, optional
        Number of rows. If `None`, will attempt to make a square grid.
    digit_kw : dict, optional
        Dictionary of keyword arguments to `plot_digit(...)`.
    **kwargs
        Keyword arguments to `plt.subplots(...)`.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, numpy.ndarray[matplotlib.axes.Axes]]
        The created figure and subplot axes.
    """
    imgs = np.asarray(imgs)
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


def plot_ellipse(x, y, angle, major, minor, ax=None, **kwargs):
    """Plots an ellipse (useful e.g. to visualise image moments).

    Parameters
    ----------
    x, y : float
        Coordinates of the centre of the ellipse.
    angle : float
        Angle of the major axis, in radians.
    major, minor : float
        Lengths of major and minor axes of the ellipse.
    ax : matplotlib.axes.Axes, optional
        Axes onto which to plot. Defaults to current axes.
    **kwargs
        Keyword arguments to `matplotlib.patches.Ellipse(...)`.
    """
    if ax is None:
        ax = plt.gca()
    ax.add_patch(Ellipse(xy=(x, y), width=2 * major, height=2 * minor,
                         angle=np.rad2deg(angle), **kwargs))


def plot_parallelogram(top_left, top_right, bottom_right, bottom_left, scale=1., ax=None, **kwargs):
    """Plots a parallelogram given its corners.

    Parameters
    ----------
    top_left, top_right, bottom_right, bottom_left : (2,) array_like
        Parallelogram corners.
    scale : float, optional
        Scaling factor, useful if plotting over an image with different resolution.
    ax : matplotlib.axes.Axes, optional
        Axes onto which to plot. Defaults to current axes.
    **kwargs
        Keyword arguments to `matplotlib.axes.Axes.plot(...)`.
    """
    if ax is None:
        ax = plt.gca()
    corners = [top_left, top_right, bottom_right, bottom_left, top_left]
    ax.plot(*(scale * np.array(corners).T - .5), **kwargs)
