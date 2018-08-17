from experiments import train_gan, train_vae

if __name__ == '__main__':
    common_kwargs = dict(
        use_cuda=True,
        ckpt_root="/data/morphomnist/checkpoints",
        data_dirs="/vol/biomedic/users/dc315/mnist/plain",
        weights=None,
        num_epochs=20,
        batch_size=64,
        save=True,
        resume=False,
        plot=False,
    )

    train_vae.main(**common_kwargs, latent_dim=64)  # VAE-64
    train_gan.main(**common_kwargs, latent_dim=64)  # GAN-64
    train_gan.main(**common_kwargs, latent_dim=2)   # GAN-2
