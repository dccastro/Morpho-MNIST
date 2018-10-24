import numpy as np
from scipy.ndimage import filters
from skimage import morphology

from .morpho import ImageMoments, ImageMorphology

_NB_MASK = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], int)


def get_angle(skel: np.ndarray, i: int, j: int, r: int) -> float:
    """Estimates the local angle of the skeleton inside a square window.

    Parameters
    ----------
    skel : np.ndarray
        Input skeleton image.
    i : int
        Vertical coordinate of the window centre.
    j : int
        Horizontal coordinate of the window centre.
    r : int
        Radius of the window.

    Returns
    -------
    float
        The estimated angle, in radians.
    """
    skel = np.pad(skel, pad_width=r, mode='constant', constant_values=0)
    mask = np.ones([2 * r + 1, 2 * r + 1])
    nbs = skel[i:i + 2*r + 1, j:j + 2*r + 1]
    angle = ImageMoments(nbs * mask).angle
    return angle


def num_neighbours(skel: np.ndarray) -> np.ndarray:
    """Computes the number of neighbours of each skeleton pixel.

    Parameters
    ----------
    skel : np.ndarray
        Input skeleton image.

    Returns
    -------
    np.ndarray
        Array containing the numbers of neighbours at each skeleton pixel and 0 elsewhere,
        with the same shape as `skel`.
    """
    skel = skel.astype(int)
    return filters.convolve(skel, _NB_MASK, mode='constant') * skel


def erase(skel: np.ndarray, seeds: np.ndarray, r: int) -> np.ndarray:
    """Erases pixels around given locations in a skeleton image.

    Parameters
    ----------
    skel : np.ndarray
        Input skeleton image.
    seeds : np.ndarray
        Locations around which to erase.
    r : int
        Radius to erase around `seeds`.

    Returns
    -------
    np.ndarray
        Processed skeleton image, of the same shape as `skel`.
    """
    erased = np.pad(skel, pad_width=r, mode='constant', constant_values=0)
    brush = ~morphology.disk(r).astype(bool)
    for i, j in zip(*np.where(seeds)):
        erased[i:i + 2*r+1, j:j + 2*r+1] &= brush
    return erased[r:-r, r:-r]


class LocationSampler(object):
    """A helper class to sample random pixel locations along an image skeleton."""

    def __init__(self, prune_tips: float = None, prune_forks: float = None):
        """
        Parameters
        ----------
        prune_tips : float, optional
            Radius to avoid around skeleton tips, in low-resolution pixel scale.
        prune_forks : float, optional
            Radius to avoid around skeleton forks, in low-resolution pixel scale.
        """
        self.prune_tips = prune_tips
        self.prune_forks = prune_forks

    def sample(self, morph: ImageMorphology, num: int = None) -> np.ndarray:
        """Samples locations along the skeleton.

        Parameters
        ----------
        morph : morphomnist.morpho.ImageMorphology
            Morphological pipeline computed for the input image.
        num : int, optional
            Number of coordinates to sample (default: one).

        Returns
        -------
        np.ndarray
            Vertical and horizontal indices of the sampled locations. If `num` is not `None`,
            points are indexed along the first axis.
        """
        skel = morph.skeleton

        if self.prune_tips is not None:
            up_prune = int(self.prune_tips * morph.scale)
            skel = erase(skel, num_neighbours(skel) == 1, up_prune)
        if self.prune_forks is not None:
            up_prune = int(self.prune_forks * morph.scale)
            skel = erase(skel, num_neighbours(skel) == 3, up_prune)

        coords = np.array(np.where(skel)).T
        if coords.shape[0] == 0:
            raise ValueError("Overpruned skeleton")
        centre_idx = np.random.choice(coords.shape[0], size=num)
        return coords[centre_idx]
