import numpy as np
import pandas as pd

from analysis import kernels, mmd

if __name__ == '__main__':
    cols = ['length', 'thickness', 'slant', 'width', 'height']
    # cols = ['width', 'height']
    test_metrics = pd.read_csv("/vol/biomedic/users/dc315/mnist/plain/t10k-morpho.csv")[cols]
    specs = ["VAE-64_plain", "GAN-64_plain", "GAN-2_plain"]
    N = 10000
    np.set_printoptions(linewidth=160, precision=4)
    seed = 123456
    print(seed)
    for spec in specs:
        print("Test data vs.", spec)
        sample_metrics = pd.read_csv(f"/data/morphomnist/metrics/{spec}_metrics.csv")[cols]

        test_metrics = test_metrics.iloc[:N]
        sample_metrics = sample_metrics.iloc[:N]

        linear = False
        unbiased = True
        bw = 'std'
        factor = 'scott'

        scale = kernels.bandwidth(test_metrics, sample_metrics, type=bw, factor=factor)
        print("Scale:", scale)
        mmd.test(test_metrics, sample_metrics, linear=True, unbiased=unbiased,
                 chunksize=None, seed=seed, scale=scale)

        # print("Mean: {:.6g}, std.: {:.6g}".format(*(lambda x: (np.mean(x), np.std(x)))(
        #     [mmd.mmd2(test_metrics, sample_metrics, linear=True, seed=seed, scale=scale)[0]
        #      for _ in range(1000)])))

        print()
