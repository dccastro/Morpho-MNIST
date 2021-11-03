import numpy as np
from skimage import draw, morphology, transform

from . import skeleton
from .morpho import ImageMorphology


class Perturbation:
    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        """Apply the perturbation.

        Parameters
        ----------
        morph : ImageMorphology
            Morphological pipeline computed for the input image.

        Returns
        -------
        (scale*H, scale*W) numpy.ndarray
            The perturbed high-resolution image. Call `morph.downscale(...)` to transform it back
            to low-resolution.
        """
        raise NotImplementedError


class Thinning(Perturbation):
    """Thin a digit by a specified proportion of its thickness."""

    def __init__(self, amount: float = .7):
        """
        Parameters
        ----------
        amount : float, optional
            Amount of thinning relative to the estimated thickness (e.g. `amount=0.7` will
            reduce the thickness by approximately 70%).
        """
        self.amount = amount

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        radius = int(self.amount * morph.scale * morph.mean_thickness / 2.)
        return morphology.erosion(morph.binary_image, morphology.disk(radius))


class Thickening(Perturbation):
    """Thicken a digit by a specified proportion of its thickness."""

    def __init__(self, amount: float = 1):
        """
        Parameters
        ----------
        amount : float, optional
            Amount of thinning relative to the estimated thickness (e.g. `amount=1.0` will
            increase the thickness by approximately 100%).
        """
        self.amount = amount

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        radius = int(self.amount * morph.scale * morph.mean_thickness / 2.)
        return morphology.dilation(morph.binary_image, morphology.disk(radius))


class Deformation(Perturbation):
    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        return transform.warp(morph.binary_image, lambda xy: self.warp(xy, morph))

    def warp(self, xy: np.ndarray, morph: ImageMorphology) -> np.ndarray:
        """Transform a regular coordinate grid to the deformed coordinates in input space.

        Parameters
        ----------
        xy : (H*W, 2) numpy.ndarray
            Regular coordinate grid in output space.
        morph : ImageMorphology
            Morphological pipeline computed for the input image.

        Returns
        -------
        (H*W, 2) numpy.ndarray
            Warped coordinates in input space.
        """
        raise NotImplementedError


class Swelling(Deformation):
    """Create a local swelling at a random location along the skeleton.

    Coordinates within `radius` :math:`R` of the centre location :math:`r_0` are warped according
    to a radial power transform: :math:`f(r) = r_0 + (r-r_0)(|r-r_0|/R)^{\gamma-1}`, where
    :math:`\gamma` is the `strength`.
    """

    def __init__(self, strength: float = 3, radius: float = 7):
        """
        Parameters
        ----------
        strength : float, optional
            Exponent of radial power transform (>1).
        radius : float, optional
            Radius to be affected by the swelling, relative to low-resolution pixel scale.
        """
        self.strength = strength
        self.radius = radius
        self.loc_sampler = skeleton.LocationSampler()

    def warp(self, xy: np.ndarray, morph: ImageMorphology) -> np.ndarray:
        centre = self.loc_sampler.sample(morph)[::-1]
        radius = (self.radius * np.sqrt(morph.mean_thickness) / 2.) * morph.scale

        offset_xy = xy - centre
        distance = np.hypot(*offset_xy.T)
        weight = (distance / radius) ** (self.strength - 1)
        weight[distance > radius] = 1.
        return centre + weight[:, None] * offset_xy


class Fracture(Perturbation):
    """Add fractures to a digit.

    Fractures are added at random locations along the skeleton, while avoiding stroke tips and
    forks, and are locally perpendicular to the pen stroke.
    """

    _ANGLE_WINDOW = 2
    _FRAC_EXTENSION = .5

    def __init__(self, thickness: float = 1.5, prune: float = 2, num_frac: int = 3):
        """
        Parameters
        ----------
        thickness : float, optional
            Thickness of the fractures, in low-resolution pixel scale.
        prune : float, optional
            Radius to avoid around stroke tips and forks, in low-resolution pixel scale.
        num_frac : int, optional
            Number of fractures to add.
        """
        self.thickness = thickness
        self.prune = prune
        self.num_frac = num_frac
        self.loc_sampler = skeleton.LocationSampler(prune, prune)

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        up_thickness = self.thickness * morph.scale
        r = int(np.ceil((up_thickness - 1) / 2))
        brush = ~morphology.disk(r).astype(bool)
        frac_img = np.pad(morph.binary_image, pad_width=r, mode='constant', constant_values=False)
        try:
            centres = self.loc_sampler.sample(morph, self.num_frac)
        except ValueError:  # Skeleton vanished with pruning, attempt without
            centres = skeleton.LocationSampler().sample(morph, self.num_frac)
        for centre in centres:
            p0, p1 = self._endpoints(morph, centre)
            self._draw_line(frac_img, p0, p1, brush)
        return frac_img[r:-r, r:-r]

    def _endpoints(self, morph, centre):
        angle = skeleton.get_angle(morph.skeleton, *centre, self._ANGLE_WINDOW * morph.scale)
        length = morph.distance_map[centre[0], centre[1]] + self._FRAC_EXTENSION * morph.scale
        angle += np.pi / 2.  # Perpendicular to the skeleton
        normal = length * np.array([np.sin(angle), np.cos(angle)])
        p0 = (centre + normal).astype(int)
        p1 = (centre - normal).astype(int)
        return p0, p1

    @staticmethod
    def _draw_line(img, p0, p1, brush):
        h, w = brush.shape
        ii, jj = draw.line(*p0, *p1)
        for i, j in zip(ii, jj):
            try:
                img[i:i + h, j:j + w] &= brush
            except ValueError:
                # Rare case: Fracture would leave image outline, because
                # selected point on skeleton is too close to image outline.
                # Ignore the fracture parts outside the image, but keep the
                # parts within the image.
                pass
