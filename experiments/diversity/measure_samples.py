import multiprocessing
import os

import models
from experiments import spec_util
from models import gan, vae
from morphomnist import measure

CHECKPOINT_ROOT = "/data/morphomnist/checkpoints"
METRICS_ROOT = "/data/morphomnist/metrics"


def main(spec, num_samples, pool):
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, spec)
    model_type, model_args, dataset_names = spec_util.parse_setup_spec(spec)
    if model_type == 'VAE':
        model = vae.VAE(model_args)
        trainer = vae.Trainer(model, beta=4.)
        trainer.cuda()
        models.load_checkpoint(trainer, checkpoint_dir)
        model.eval()
        sample_latent = model.sample_latent(num_samples)
        sample_imgs = model.dec(sample_latent)
    elif model_type in ['GAN', 'GANmc']:
        model = gan.GAN(model_args)
        trainer = gan.Trainer(model)
        trainer.cuda()
        models.load_checkpoint(trainer, checkpoint_dir)
        model.eval()
        sample_imgs = model(num_samples)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    print(f"Loaded model {checkpoint_dir}. Measuring samples...")
    sample_imgs_np = sample_imgs.detach().cpu().squeeze().numpy()
    sample_metrics = measure.measure_batch(sample_imgs_np, pool=pool)

    os.makedirs(METRICS_ROOT, exist_ok=True)
    metrics_path = os.path.join(METRICS_ROOT, f"{spec}_metrics.csv")
    sample_metrics.to_csv(metrics_path, index_label='index')
    print(f"Morphometrics saved to {metrics_path}")


if __name__ == '__main__':
    specs = [
        "VAE-64_plain",
        "GAN-64_plain",
        "GAN-2_plain",
        "GAN-1_plain",
    ]
    N_sample = 10000
    with multiprocessing.Pool() as pool:
        for spec in specs:
            main(spec, N_sample, pool)
