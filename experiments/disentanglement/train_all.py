import os

from experiments import train_infogan

if __name__ == '__main__':
    common_kwargs = dict(
        use_cuda=True,
        ckpt_root="/data/morphomnist/checkpoints/weighted",
        num_epochs=20,
        batch_size=64,
        save=True,
        resume=False,
        plot=False,
        cat_dim=10,
        noise_dim=62,
    )
    data_root = "/vol/biomedic/users/dc315/mnist"
    # train_infogan.main(**common_kwargs, cont_dim=2, bin_dim=0,
    #     data_dirs=os.path.join(data_root, "plain"), weights=None)
    #
    # train_infogan.main(**common_kwargs, cont_dim=3, bin_dim=0,
    #     data_dirs=[os.path.join(data_root, name) for name in ["plain", "pert-thin-thic"]],
    #     weights=[1, 2])
    #
    # train_infogan.main(**common_kwargs, cont_dim=3, bin_dim=0,
    #     data_dirs=[os.path.join(data_root, name) for name in ["plain", "pert-swel-frac"]],
    #     weights=[1, 2])

    train_infogan.main(**common_kwargs, cont_dim=2, bin_dim=2,
                       data_dirs=[os.path.join(data_root, name)
                                  for name in ["plain", "pert-swel-frac"]],
                       weights=[1, 2])
