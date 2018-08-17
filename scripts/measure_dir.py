import argparse
import multiprocessing
import os

from morphomnist import idx, measure


def measure_dir(data_dir, pool):
    for name in ['t10k', 'train']:
        in_path = os.path.join(data_dir, name + "-images-idx3-ubyte.gz")
        out_path = os.path.join(data_dir, name + "-morpho.csv")
        print(f"Processing MNIST data file {in_path}...")
        data = idx.load(in_path)
        df = measure.measure_batch(data, pool=pool, chunksize=100)
        df.to_csv(out_path, index_label='index')
        print(f"Morphometrics saved to {out_path}")


def main(data_dirs):
    with multiprocessing.Pool() as pool:
        for data_dir in data_dirs:
            measure_dir(data_dir, pool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datadirs', nargs='+')
    args = parser.parse_args()
    print(args.datadirs)

    assert all(os.path.exists(data_dir) for data_dir in args.datadirs)

    main(args.datadirs)
