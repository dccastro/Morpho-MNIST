import numpy as np
from skimage import draw, morphology, transform

from . import skeleton
from .morpho import ImageMoments, ImageMorphology, bounding_parallelogram


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
        h_start, w_start = h // 2, w // 2
        h_end, w_end = h - h_start, w - w_start
        ii, jj = draw.line(*p0, *p1)
        for i, j in zip(ii, jj):
            try:
                img[i - h_start:i + h_end, j - w_start:j + w_end] &= brush
            except ValueError:
                # Rare case: Fracture would leave image outline, because
                # selected point on skeleton is too close to image outline.
                # Ignore the fracture parts outside the image, but keep the
                # parts within the image.
                pass


def _get_disk(radius: int, scale: int) -> np.ndarray:
    mag_radius = scale * radius
    mag_disk = morphology.disk(mag_radius, dtype=np.float64)
    disk = transform.pyramid_reduce(mag_disk, downscale=scale, order=1, channel_axis=None)
    return disk  # type: ignore


class SetThickness(Perturbation):
    _disk_cache: dict[int, np.ndarray] = {}

    def __init__(self, target_thickness: float):
        self.target_thickness = target_thickness

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        delta = self.target_thickness - morph.mean_thickness
        radius = int(morph.scale * abs(delta) / 2.)
        if radius in self._disk_cache:
            disk = self._disk_cache[radius]
        else:
            disk = _get_disk(radius, scale=16)
            self._disk_cache[radius] = disk
        img = morph.binary_image
        if delta >= 0:
            return morphology.dilation(img, disk)
        else:
            return morphology.erosion(img, disk)


class LinearDeformation(Deformation):
    def _get_matrix(self, moments: ImageMoments, morph: ImageMorphology) -> np.ndarray:
        raise NotImplementedError

    def warp(self, xy: np.ndarray, morph: ImageMorphology) -> np.ndarray:
        moments = ImageMoments(morph.binary_image)
        centroid = np.array(moments.centroid)
        matrix = self._get_matrix(moments, morph)
        xy_ = (xy - centroid) @ matrix.T + centroid
        return xy_


class SetSlant(LinearDeformation):
    def __init__(self, target_slant_rad: float):
        self.target_shear = -np.tan(target_slant_rad)

    def _get_matrix(self, moments: ImageMoments, morph: ImageMorphology) -> np.ndarray:
        source_shear = moments.horizontal_shear
        delta = self.target_shear - source_shear
        return np.array([[1., -delta], [0., 1.]])


def _measure_width(morph: ImageMorphology, frac=.02, moments: ImageMoments | None = None) -> float:
    top_left, top_right = bounding_parallelogram(morph.hires_image,
                                                 frac=frac, moments=moments)[:2]
    return (top_right[0] - top_left[0]) / morph.scale


class SetWidth(LinearDeformation):
    _tolerance = 1.

    def __init__(self, target_width: float, validate=False):
        self.target_width = target_width
        self._validate = validate

    def _get_matrix(self, moments: ImageMoments, morph: ImageMorphology) -> np.ndarray:
        source_width = _measure_width(morph, moments=moments)
        factor = source_width / self.target_width
        shear = moments.horizontal_shear
        return np.array([[factor, shear * (1. - factor)], [0., 1.]])

    def __call__(self, morph: ImageMorphology) -> np.ndarray:
        pert_hires_image = super().__call__(morph)
        if self._validate:
            pert_image = morph.downscale(pert_hires_image)
            pert_morph = ImageMorphology(pert_image, threshold=morph.threshold, scale=morph.scale)
            width = _measure_width(pert_morph)
            if abs(width - self.target_width) > self._tolerance:
                print(f"Incorrect width after transformation: {width:.1f}, "
                      f"expected {self.target_width:.1f}.")
                pert_hires_image = self(pert_morph)
        return pert_hires_image
