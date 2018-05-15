import multiprocessing
import os

import morpho
import util

if __name__ == '__main__':
    until = 10
    root = "../data/mnist"
    with multiprocessing.Pool() as pool:
        for folder in ['orig', 'patho']:
            for name in ['t10k', 'train']:
                data = util.load(os.path.join(root, folder, name + "-images-idx3-ubyte.gz"))[:until]
                df = morpho.measure_batch(data, pool=pool, chunksize=1000)
                df.to_csv(os.path.join(root, folder, name + "-morpho.csv"), index_label='index')
