import numpy as np
import pytest

from morphomnist import perturb
from morphomnist.measure import Morphometrics, measure_image
from morphomnist.morpho import ImageMoments, ImageMorphology


@pytest.fixture
def real_image() -> np.ndarray:
    return np.load("tests/train-image-00000.npy")


@pytest.fixture
def real_morphology(real_image: np.ndarray) -> ImageMorphology:
    return ImageMorphology(real_image, scale=4)


class MockDeformation(perturb.Deformation):
    """Mock identity deformation to test the call to `skimage.transform.warp`."""
    def warp(self, xy: np.ndarray, morph: ImageMorphology) -> np.ndarray:
        return xy


class MockLinearDeformation(perturb.LinearDeformation):
    """Mock identity deformation to test the call to `Deformation.warp`."""
    def _get_matrix(self, moments: ImageMoments, morph: ImageMorphology) -> np.ndarray:
        return np.eye(2)


@pytest.mark.parametrize("perturbation", [
    perturb.Thinning(),
    perturb.Thickening(),
    perturb.Swelling(),
    perturb.Fracture(),
    perturb.SetThickness(4),
    perturb.SetSlant(0.5),
    perturb.SetWidth(10),
    MockDeformation(),
    MockLinearDeformation(),
])
def test_perturbation(
    real_image: np.ndarray, real_morphology: ImageMorphology, perturbation: perturb.Perturbation
) -> None:
    perturbed_hires = perturbation(real_morphology)
    perturbed = real_morphology.downscale(perturbed_hires)
    assert perturbed.shape == real_image.shape
    assert perturbed.dtype == real_image.dtype
    assert np.all(perturbed >= real_image.min())
    assert np.all(perturbed <= real_image.max())

    abs_diff = np.abs(perturbed - real_image).mean()
    rel_diff = abs_diff / real_image.ptp()
    assert rel_diff < 0.2


def perturb_and_measure(real_morphology: ImageMorphology, perturbation: perturb.Perturbation) -> Morphometrics:
    perturbed_hires = perturbation(real_morphology)
    perturbed = real_morphology.downscale(perturbed_hires)
    metrics = measure_image(perturbed)
    return metrics


def test_set_thickness(real_morphology: ImageMorphology) -> None:
    target_thickness = 4
    metrics = perturb_and_measure(real_morphology, perturb.SetThickness(target_thickness))
    assert np.isclose(metrics.thickness, target_thickness, atol=0.2)


def test_set_slant(real_morphology: ImageMorphology) -> None:
    target_slant_rad = -0.5
    metrics = perturb_and_measure(real_morphology, perturb.SetSlant(target_slant_rad))
    assert np.isclose(metrics.slant, target_slant_rad, atol=0.01)


def test_set_width(real_morphology: ImageMorphology) -> None:
    target_width = 20
    metrics = perturb_and_measure(real_morphology, perturb.SetWidth(target_width))
    assert np.isclose(metrics.width, target_width, atol=1)
