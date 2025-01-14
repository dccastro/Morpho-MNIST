import numpy as np
import pytest

from morphomnist.morpho import ImageMorphology
from morphomnist.perturb import Deformation, Fracture, Perturbation, Swelling, Thickening, Thinning


@pytest.fixture
def real_image() -> np.ndarray:
    return np.load("tests/train-image-00000.npy")


@pytest.fixture
def real_morphology(real_image: np.ndarray) -> ImageMorphology:
    return ImageMorphology(real_image)


class MockDeformation(Deformation):
    """Mock identity deformation to test the call to `skimage.transform.warp`."""
    def warp(self, xy: np.ndarray, morph: ImageMorphology) -> np.ndarray:
        return xy


@pytest.mark.parametrize("perturbation", [
    Thinning(),
    Thickening(),
    Swelling(),
    Fracture(),
    MockDeformation(),
])
def test_perturbation(
    real_image: np.ndarray, real_morphology: ImageMorphology, perturbation: Perturbation
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
