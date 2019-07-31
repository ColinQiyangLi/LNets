echo "BCOP mnist"
srun --gres=gpu:1 -x gpu027 --mem=12G --partition=p100 python -u fid_score.py /h/anilcem/Workspace/architectures/lnets_for_conv/LNets/out/gan/LGANS/BCOP/GAN_training_LWGAN_MNIST_mnist_LWGAN_2019_07_29_12_44_09_444534/saved_samples/generated /h/anilcem/Workspace/architectures/lnets_for_conv/LNets/out/gan/LGANS/BCOP/GAN_training_LWGAN_MNIST_mnist_LWGAN_2019_07_29_12_44_09_444534/saved_samples/real --gpu 0 > BCOP_mnist.txt &

echo "BCOP cifar10"
srun --gres=gpu:1 -x gpu027 --mem=12G --partition=p100 python -u fid_score.py /h/anilcem/Workspace/architectures/lnets_for_conv/LNets/out/gan/LGANS/BCOP/GAN_training_LWGAN_cifar10_LWGAN_2019_07_28_22_12_33_490158/saved_samples/generated /h/anilcem/Workspace/architectures/lnets_for_conv/LNets/out/gan/LGANS/BCOP/GAN_training_LWGAN_cifar10_LWGAN_2019_07_28_22_12_33_490158/saved_samples/real --gpu 0 > BCOP_cifar10.txt &

echo "RKO mnist"
srun --gres=gpu:1 -x gpu027 --mem=12G --partition=p100 python -u fid_score.py /h/anilcem/Workspace/architectures/lnets_for_conv/LNets/out/gan/LGANS/RKO/GAN_training_LWGAN_MNIST_mnist_LWGAN_2019_07_29_12_44_38_052832/saved_samples/generated /h/anilcem/Workspace/architectures/lnets_for_conv/LNets/out/gan/LGANS/RKO/GAN_training_LWGAN_MNIST_mnist_LWGAN_2019_07_29_12_44_38_052832/saved_samples/real --gpu 0 > RKO_mnist.txt &

echo "RKO cifar10"
srun --gres=gpu:1 -x gpu027 --mem=12G --partition=p100 python -u fid_score.py /h/anilcem/Workspace/architectures/lnets_for_conv/LNets/out/gan/LGANS/RKO/GAN_training_LWGAN_cifar10_LWGAN_2019_07_28_22_13_11_289824/saved_samples/generated /h/anilcem/Workspace/architectures/lnets_for_conv/LNets/out/gan/LGANS/RKO/GAN_training_LWGAN_cifar10_LWGAN_2019_07_28_22_13_11_289824/saved_samples/real --gpu 0 > RKO_cifar10.txt &
