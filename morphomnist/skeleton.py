from collections import deque

import numpy as np
from scipy.ndimage import filters
from skimage import morphology

from .morpho import ImageMoments, ImageMorphology

_NB_MASK = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], int)


def get_angle(skel, i, j, r):
    skel = np.pad(skel, pad_width=r, mode='constant', constant_values=0)
    mask = np.ones([2 * r + 1, 2 * r + 1])
    nbs = skel[i:i + 2*r + 1, j:j + 2*r + 1]
    angle = ImageMoments(nbs * mask).angle
    return angle


def prune(skel, num_iter):
    skel = skel.astype(int)
    for i in range(num_iter):
        corr = filters.convolve(skel, _NB_MASK, mode='constant')
        skel[corr == 1] = 0
    return skel


def num_neighbours(skel):
    skel = skel.astype(int)
    return filters.convolve(skel, _NB_MASK, mode='constant') * skel


def skeleton_distance(skel, seeds):
    skel = skel.astype(int)
    nbs = np.where(_NB_MASK)
    distance = np.ones(skel.shape) * np.hypot(*skel.shape)

    q = deque()
    roots = {}
    for i, j in zip(*np.where(seeds)):
        distance[i, j] = 0
        roots[i, j] = i, j
        q.appendleft((i, j))
    while q:
        i0, j0 = q.pop()
        for di, dj in zip(*nbs):
            i = i0 + di - 1
            j = j0 + dj - 1
            if (0 <= i < skel.shape[0]) and (0 <= j < skel.shape[1]) and skel[i, j]:
                dist = np.hypot(roots[i0, j0][0] - i, roots[i0, j0][1] - j)
                if dist < distance[i, j]:
                    distance[i, j] = dist
                    roots[i, j] = roots[i0, j0]
                    q.append((i, j))
    return distance * skel


def erase(skel, seeds, r):
    erased = np.pad(skel, pad_width=r, mode='constant', constant_values=0)
    brush = ~morphology.disk(r).astype(bool)
    for i, j in zip(*np.where(seeds)):
        erased[i:i + 2*r+1, j:j + 2*r+1] &= brush
    return erased[r:-r, r:-r]


class LocationSampler(object):
    def __init__(self, prune_tips: float = None, prune_forks: float = None):
        self.prune_tips = prune_tips
        self.prune_forks = prune_forks

    def sample(self, morph: ImageMorphology, num: int = None):
        skel = morph.skeleton

        if self.prune_tips is not None:
            up_prune = int(self.prune_tips * morph.scale)
            skel = erase(skel, num_neighbours(skel) == 1, up_prune)
        if self.prune_forks is not None:
            up_prune = int(self.prune_tips * morph.scale)
            skel = erase(skel, num_neighbours(skel) == 3, up_prune)

        coords = np.array(np.where(skel)).T
        centre_idx = np.random.choice(coords.shape[0], size=num)
        return coords[centre_idx]
