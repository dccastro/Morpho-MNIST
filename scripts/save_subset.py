import argparse
from pathlib import Path

from morphomnist import io


def main(raw_dir: Path, out_dir: Path, num_samples: int = 3, start: int = 0):
    images_filename = "train-images-idx3-ubyte.gz"
    labels_filename = "train-labels-idx1-ubyte.gz"

    indices = slice(start, start + num_samples)
    range_str = f"{start}_{start + num_samples}"

    images_subset_filename = f"train-images-{range_str}-idx3-ubyte.gz"
    labels_subset_filename = f"train-labels-{range_str}-idx1-ubyte.gz"

    all_images = io.load_idx(str(raw_dir / images_filename))
    io.save_idx(all_images[indices], str(out_dir / images_subset_filename))
    print(f"Saved {num_samples} images to {out_dir / images_subset_filename}")

    all_labels = io.load_idx(str(raw_dir / labels_filename))
    io.save_idx(all_labels[indices], str(out_dir / labels_subset_filename))
    print(f"Saved {num_samples} labels to {out_dir / labels_subset_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save a subset of MNIST images and labels.")
    parser.add_argument("raw_dir", type=Path, help="Directory containing the raw MNIST data")
    parser.add_argument("out_dir", type=Path, help="Output directory for the subset data")
    parser.add_argument("--num", type=int, default=3, help="Number of samples to save")
    parser.add_argument("--start", type=int, default=0, help="Index of the first image to save")
    args = parser.parse_args()

    main(
        raw_dir=args.raw_dir.expanduser(),
        out_dir=args.out_dir.expanduser(),
        num_samples=args.num,
        start=args.start,
    )
