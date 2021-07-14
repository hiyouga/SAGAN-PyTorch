import torch
import torch.nn as nn
from sagan_models import Generator, Discriminator


class SAGAN_Trainer:

    def __init__(self, args):
        self.args = args
        self._build_model()

    def _build_model(self):
        self.G = Generator(self.args.batch_size, self.args.im_size, self.args.z_dim, self.args.g_conv_dim, self.args.adv_loss)
        self.D = Discriminator(self.args.batch_size, self.args.im_size, self.args.d_conv_dim, self.args.adv_loss)
        self.G.to(self.args.device)
        self.D.to(self.args.device)
        if self.args.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        _G_params = filter(lambda p: p.requires_grad, self.G.parameters())
        _D_params = filter(lambda p: p.requires_grad, self.D.parameters())
        self.g_optimizer = torch.optim.Adam(_G_params, self.args.g_lr, [self.args.beta1, self.args.beta2])
        self.d_optimizer = torch.optim.Adam(_D_params, self.args.d_lr, [self.args.beta1, self.args.beta2])
        self.c_loss = nn.CrossEntropyLoss()

    def train(self, real_images):
        ''' train discriminator '''
        d_out_real = self.D(real_images)
        ''' compute loss with real images '''
        if self.args.adv_loss == 'wgan-gp':
            d_loss_real = -1 * torch.mean(d_out_real)
        elif self.args.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        ''' apply Gumbel Softmax '''
        z = torch.randn(real_images.size(0), self.args.z_dim).to(self.args.device)
        fake_images = self.G(z)
        d_out_fake = self.D(fake_images)
        ''' compute loss with fake images '''
        if self.args.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.args.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        ''' update discriminator '''
        d_loss = d_loss_real + d_loss_fake
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        if self.args.adv_loss == 'wgan-gp':
            ''' compute gradient penalty '''
            alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.args.device)
            interpolated = alpha * real_images.data + (1 - alpha) * fake_images.data
            interpolated.requires_grad_(True)
            d_out_itp = self.D(interpolated)
            grad = torch.autograd.grad(outputs=d_out_itp,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(d_out_itp.size()).to(self.args.device),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]
            grad_norm = grad.view(grad.size(0), -1).norm(dim=1)
            d_loss_gp = torch.mean((grad_norm - 1) ** 2)
            ''' optimize gradient penalty '''
            d_loss = self.args.lambda_gp * d_loss_gp
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()
        ''' train generator '''
        z = torch.randn(real_images.size(0), self.args.z_dim).to(self.args.device)
        fake_images = self.G(z)
        g_out_fake = self.D(fake_images)
        ''' compute loss with fake images '''
        g_loss_fake = -1 * g_out_fake.mean()
        self.g_optimizer.zero_grad()
        g_loss_fake.backward()
        self.g_optimizer.step()
        return d_loss_real.item(), d_loss_fake.item(), g_loss_fake.item()
