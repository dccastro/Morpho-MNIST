import struct

import numpy as np


def load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def save_uint8(data, f):
    f.write(struct.pack('BBBB', 0, 0, 0x08, data.ndim))
    f.write(struct.pack('>' + 'I' * data.ndim, *data.shape))
    f.write(data.tobytes())


if __name__ == '__main__':
    # MNIST_FILE = "data/mnist/raw/train-images-idx3-ubyte"
    MNIST_FILE = "data/mnist/patho/train-images-idx3-ubyte"
    TEST_FILE = "test"

    with open(MNIST_FILE, 'rb') as f:
        data = load_uint8(f)

    with open(TEST_FILE, 'wb') as f:
        save_uint8(data, f)

    with open(TEST_FILE, 'rb') as f:
        data_ = load_uint8(f)

    import os
    os.remove(TEST_FILE)

    print(data.shape)
    print(data_.shape)

    import matplotlib.pyplot as plt
    import util

    for i in np.random.permutation(len(data)):
        util.plot_digit(data[i], plt.subplot(121))
        util.plot_digit(data_[i], plt.subplot(122))
        plt.suptitle("Digit {} / {}".format(i, len(data)))
        plt.show()
