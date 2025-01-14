import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from multiprocessing import Pool

from morphomnist.measure import measure_image, measure_batch, Morphometrics


@pytest.fixture
def real_image() -> np.ndarray:
    return np.load("tests/train-image-00000.npy")


@pytest.fixture
def real_morphometrics() -> Morphometrics:
    return Morphometrics(  # From the released data: original/train-morpho.csv
        area=107.3125,
        length=50.26650429449552,
        thickness=2.4606583773939543,
        slant=0.23107446620980754,
        width=14.539571782767355,
        height=19.849053691610763,
    )


def test_measure_image(real_image: np.ndarray, real_morphometrics: Morphometrics) -> None:
    result = measure_image(real_image, verbose=False)
    npt.assert_allclose(result, real_morphometrics, rtol=0.5)


def test_measure_batch(real_image: np.ndarray) -> None:
    images = np.tile(real_image, (5, 1, 1))
    single_result = measure_image(real_image, verbose=False)
    batch_result = measure_batch(images, pool=None)
    expected = pd.DataFrame([single_result] * 5)
    pd.testing.assert_frame_equal(batch_result, expected)


def test_measure_batch_parallel(real_image: np.ndarray) -> None:
    images = np.tile(real_image, (5, 1, 1))
    serial_result = measure_batch(images, pool=None)
    with Pool(2) as pool:
        parallel_result = measure_batch(images, pool=pool, chunksize=2)
    pd.testing.assert_frame_equal(parallel_result, serial_result)
