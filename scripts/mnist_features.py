import multiprocessing
import timeit

import pandas as pd
import torchvision

from morpho import ImageMoments, ImageMorphology, bounding_parallelogram

DATA_ROOT = "../data/mnist"
THRESHOLD = 128
UP_FACTOR = 8
COLUMNS = ['digit', 'area', 'length', 'mean_thck', 'median_thck', 'shear', 'width', 'height']


def morpho_features(morph: ImageMorphology, moments: ImageMoments, verbose=True):
    mean_skel_thck = morph.mean_thickness
    median_skel_thck = morph.median_thickness
    area = morph.area
    length = morph.stroke_length
    shear = moments.horizontal_shear

    corners = bounding_parallelogram(morph.hires_image, .02, moments)
    width = (corners[1][0] - corners[0][0]) / morph.scale
    height = (corners[-1][1] - corners[0][1]) / morph.scale

    if verbose:
        print("Mean thickness:", mean_skel_thck)
        print("Median thickness:", mean_skel_thck)
        print("Length:", length)
        print("Dimensions: {:.1f} x {:.1f}".format(width, height))

    return area, length, mean_skel_thck, median_skel_thck, shear, width, height


def process_img(index, img_tensor, label, tag):
    start = timeit.default_timer()
    img = img_tensor.squeeze().numpy()
    morph = ImageMorphology(img, THRESHOLD, UP_FACTOR)
    moments = ImageMoments(morph.hires_image)
    results = morpho_features(morph, moments, verbose=False)
    end = timeit.default_timer()
    print("[{}-{:05d}] {:.1f} ms".format(tag, index, 1000 * (end - start)))
    return [label, *results]


def process_dataset(data, labels, pool, tag):
    args = ((i, img, label, tag) for i, (img, label) in enumerate(zip(data, labels)))
    start = timeit.default_timer()
    df = pd.DataFrame(pool.starmap(process_img, args, chunksize=1000), columns=COLUMNS)
    end = timeit.default_timer()
    print("[{}] Total time: {:.1f} s ({:.1f} ms/image)".format(
            tag, end - start, 1000 * (end - start) / len(data)))
    print(df.tail())
    return df


if __name__ == '__main__':
    transf = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, transform=transf)
    test_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, transform=transf)

    until = None
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        df_train = process_dataset(
                train_set.train_data[:until], train_set.train_labels[:until], pool, "TRAIN")
        df_train.to_csv("../mnist_features_x8_train.csv", index_label='index')

        df_test = process_dataset(
                test_set.test_data[:until], test_set.test_labels[:until], pool, "TEST")
        df_test.to_csv("../mnist_features_x8_test.csv", index_label='index')
