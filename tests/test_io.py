import filecmp
from pathlib import Path

import numpy as np

from morphomnist import io

SOURCE_DATA_FILE = "tests/train-images-0_3-idx3-ubyte.gz"


def test_load_save_idx(tmp_path: Path) -> None:
    test_data_file = str(tmp_path / "dummy-idx3-ubyte.gz")
    data = io.load_idx(SOURCE_DATA_FILE)

    io.save_idx(data, test_data_file)
    filecmp.cmp(SOURCE_DATA_FILE, test_data_file, shallow=False)

    reloaded_data = io.load_idx(test_data_file)
    assert np.alltrue(data == reloaded_data)
