import torch
import torch.utils.data
from torch.optim import Adam

from torch import nn
from torch.nn import functional as f


class UsualVAE(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.optimizer = Adam(params=list(self.vae.parameters()))

    @staticmethod
    def _loss(recon_x, x, mu, logvar):
        bce = f.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return bce + kld

    def training_step(self, x, *args, **kwargs):
        recon_batch, mu, logvar = self.vae.forward(x)
        loss = self._loss(recon_batch, x, mu, logvar)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    # we want to expose the vae internal functions
    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)

    def __call__(self, x):
        return self.vae(x)
