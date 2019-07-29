"""
This GAN implementation is a replica of the the WGAN implementation that can be found in the same directory.
The only differences are that every layer in the discriminator is replaced with its strictly Lipschitz counterpart
(the original ones are commented out for readability) and weight clipping is removed, as the resulting model will be
strictly Lipschitz already.
"""


import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim

import lnets.tasks.gan.gan_utils as utils
from lnets.tasks.gan.data.data_loader import dataloader
from lnets.models.activations import *
from lnets.models.layers import *
from munch import Munch

from lconvnet.layers.orthogonal.cleanup import LipschitzConv2d, BCOP, RKO

CUDA = False

def bcop(in_channels, out_channels, kernel_size, stride, padding, conv_module, zero_padding):
    return LipschitzConv2d(in_channels, out_channels, kernel_size, stride, padding, conv_module=conv_module, zero_padding=zero_padding)

class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x


class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, cuda=False, conv_type="default"):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        config = Munch.fromDict(dict(cuda=cuda,
                                     model=dict(linear=dict(bjorck_beta=0.5,
                                                            bjorck_iter=12,
                                                            bjorck_order=1,
                                                            safe_scaling=True
                                                            ))))

        zero_padding = True
        module = BCOP if conv_type == "BCOP" else RKO
        self.conv = nn.Sequential(
            # Conv.
            (bcop(self.input_dim, 64, 4, 2, 1, module, zero_padding) if conv_type != "default" else BjorckConv2d(self.input_dim, 64, 4, 2, 1, config=config)),

            # Activ.
            MaxMin(num_units=32, axis=1),

            # Conv
            (bcop(64, 128, 4, 2, 1, module, zero_padding) if conv_type != "default" else BjorckConv2d(64, 128, 4, 2, 1, config=config)),

            # Activ.
            MaxMin(num_units=64, axis=1)
        )
        self.fc = nn.Sequential(
            # Linear.
            BjorckLinear(128 * (self.input_size // 4) * (self.input_size // 4), 1024, config=config),

            # Activ.
            MaxMin(num_units=512),

            # Linear.
            BjorckLinear(1024, self.output_dim, config=config),
            # nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


class LWGAN(object):
    def __init__(self, args):
        # Parameters.
        self.model_dir = args.model_dir
        self.figures_dir = args.figures_dir
        self.log_dir = args.log_dir
        self.data_root = args.data_root

        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        self.c = 0.01  # Clipping value.
        self.n_critic = 5  # The number of iterations of the critic per generator iteration.

        # Load dataset.
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size, self.data_root)
        data = self.data_loader.__iter__().__next__()[0]

        # Networks initialization.
        self.G = Generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = Discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, cuda=self.gpu_mode, conv_type=args.conv_type)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # Fixed noise.
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = dict()
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # Update D network.
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = -torch.mean(D_real)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(D_fake)

                D_loss = D_real_loss + D_fake_loss

                D_loss.backward()
                self.D_optimizer.step()

                # # Clipping D.
                # Discriminator is Lipschitz already - no need to clip.

                if ((iter + 1) % self.n_critic) == 0:
                    # Update G network.
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = -torch.mean(D_fake)
                    self.train_hist['G_loss'].append(G_loss.item())

                    G_loss.backward()
                    self.G_optimizer.step()

                    self.train_hist['D_loss'].append(D_loss.item())

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size,
                           D_loss.item(), G_loss.item()))
                
                if ((iter + 1) % 800) == 0:
                    print("Saving model. ")
                    self.save()

                if ((iter + 1) % 10000) == 0:
                    self.save()

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch + 1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(os.path.join(self.figures_dir, self.model_name), self.epoch)
        utils.loss_plot(self.train_hist, self.log_dir, self.model_name)

    def get_generated(self, batch_size):
        assert batch_size == self.batch_size, "Currently batch_size can only be set during construction. "
        # Sample noise.
        z = torch.rand((batch_size, self.z_dim))
        if self.gpu_mode:
            z = z.cuda()

        # Run through the generator network.
        generated = self.G(z)

        return generated

    def get_real(self, batch_size):
        assert batch_size == self.batch_size, "Currently batch_size can only be set during construction. "

        real = self.data_loader.__iter__().__next__()[0]

        return real

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          os.path.join(self.figures_dir, self.model_name + '_epoch%03d' % epoch + '.png'))

    def save(self):
        # Save model.
        torch.save(self.G.state_dict(), os.path.join(self.model_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.model_dir, self.model_name + '_D.pkl'))

        # Save the logs.
        with open(os.path.join(self.log_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        self.G.load_state_dict(torch.load(os.path.join(self.model_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(self.model_dir, self.model_name + '_D.pkl')))
