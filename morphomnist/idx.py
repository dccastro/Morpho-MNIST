# TODO: Add documentation
import gzip
import struct

import numpy as np


def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def _save_uint8(data, f):
    if data.dtype is not np.uint8:
        data = data.astype(np.uint8)
    f.write(struct.pack('BBBB', 0, 0, 0x08, data.ndim))
    f.write(struct.pack('>' + 'I' * data.ndim, *data.shape))
    f.write(data.tobytes())


def save(data, path):
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'wb') as f:
        _save_uint8(data, f)


def load(path):
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)


if __name__ == '__main__':
    # MNIST_FILE = "data/mnist/raw/train-images-idx3-ubyte"
    MNIST_FILE = "/vol/biomedic/users/dc315/mnist/raw/train-images-idx3-ubyte.gz"
    TEST_FILE = "test"

    data = load(MNIST_FILE)
    save(data, TEST_FILE)
    data_ = load(TEST_FILE)

    import os
    os.remove(TEST_FILE)

    assert np.alltrue(data == data_)
