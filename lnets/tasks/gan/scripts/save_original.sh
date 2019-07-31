srun --gres=gpu:1 -x gpu027 --mem=12G --partition=p100 python -um lnets.tasks.gan.mains.save_gan_samples ./lnets/tasks/gan/configs/save_gan_mnist_original.json

srun --gres=gpu:1 -x gpu027 --mem=12G --partition=p100 python -um lnets.tasks.gan.mains.save_gan_samples ./lnets/tasks/gan/configs/save_gan_cifar10_original.json
