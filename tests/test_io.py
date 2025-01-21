import filecmp
from pathlib import Path

import numpy as np
import pytest

from morphomnist import io


@pytest.mark.parametrize("source_data_filename", [
    "train-images-0_3-idx3-ubyte.gz",
    "train-labels-0_3-idx1-ubyte.gz",
])
def test_load_save_idx(source_data_filename: str, tmp_path: Path) -> None:
    source_data_path = "tests/" + source_data_filename
    test_data_path = str(tmp_path / source_data_filename)
    data = io.load_idx(source_data_path)

    io.save_idx(data, test_data_path)
    filecmp.cmp(source_data_path, test_data_path, shallow=False)

    reloaded_data = io.load_idx(test_data_path)
    assert np.alltrue(data == reloaded_data)
