import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from torch.autograd import Variable


def plot_digit(x, ax=None, title=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if isinstance(x, Variable):
        x = x.data
    ax.imshow(x.squeeze(), cmap='gray_r', **kwargs)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if title is not None:
        ax.set_title(title)


def plot_ellipse(x, y, angle, major, minor, ax, **kwargs):
    ax.add_patch(Ellipse(xy=(x, y), width=2 * major, height=2 * minor, angle=np.rad2deg(angle),
                         **kwargs))
