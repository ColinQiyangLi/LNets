import numpy as np

from lnets.utils.config import process_config
from lnets.tasks.dualnets.distrib.gan_sampler import GANSampler


def collect_images(sampler, num_imgs, im_size, num_channels, sample_size):
    # Collect images.
    sampled_images = np.zeros((num_imgs, im_size, im_size, num_channels))

    count = 0
    while count < num_imgs - sample_size:
        curr_imgs = sampler(cfg.distrib1.sample_size).detach().cpu().numpy().transpose((0, 2, 3, 1))
        assert curr_imgs.shape[0] == sample_size, "Doens't match sample size, count: {}".format(count)
        sampled_images[count:count + sample_size] = curr_imgs

        count += sample_size

    # Add the last bit.
    last_samples = gan_sampler(cfg.distrib1.sample_size).detach().cpu().numpy().transpose((0, 2, 3, 1))
    sampled_images[count:] = last_samples[num_imgs - count]

    return sampled_images


def save_images(imgs):
    for i in range(imgs.shape[0]):
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    # Parse the config.
    cfg = process_config()

    # Quick checks.
    assert cfg.distrib1.generate_type == "generated"

    # Load the gan loader. 
    gan_sampler = GANSampler(cfg.distrib1)

    # Collect images.
    images = collect_images(gan_sampler, cfg.num_imgs, cfg.im_size, cfg.num_channels, cfg.distrib1.sample_size)

    # Save the images.
    save_images(images)
