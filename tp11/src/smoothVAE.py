import torch
import torch.utils.data
from torch.optim import Adam

from torch import nn
import numpy as np

from utils.projected_gradient_descent import projected_gradient_descent


def wd22(mu_a, mu_b, sigma_a, sigma_b):
    # Computes the squared Wasserstein distance between N(mu_a, sigma_a), N(mu_b, sigma_b), assuming that
    # sigma_a, b = Diag(sigma_a,b), see appendix B1
    return torch.sum(torch.pow(mu_a - mu_b, 2), dim=1) + torch.sum(
        (sigma_a + sigma_b - 2 * torch.sqrt(sigma_a * sigma_b)), dim=1)


def entropy_regularized_wd2(mu_a, mu_b, sigma_a, sigma_b, gamma):
    # See appendix C
    # This is WD_2,gamma

    a = (gamma / 2) * torch.sum(torch.pow(mu_a - mu_b, 2) + sigma_a + sigma_b, dim=1)

    b = - (1 / 2) * torch.sum(
        (u(sigma_a, sigma_b, gamma) - 1) - torch.log(u(sigma_a, sigma_b, gamma) + 1) + torch.log(2 * sigma_a * sigma_b),
        dim=1)

    c = - mu_a.shape[0] * np.log(2 * np.pi) - mu_a.shape[0]  # constant

    return a + b + c


def u(a, b, gamma):
    return torch.sqrt(1 + 4 * gamma ** 2 * a * b)


class SmoothVAE:
    def __init__(self, vae, v, gamma, ball_radius=1, norm_fn="inf", alpha=0.001, steps=None, eps=None,
                 device=torch.device('cpu')):
        super(SmoothVAE, self).__init__()

        self.device = device
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.norm_fn = norm_fn
        self.ball_radius = ball_radius
        self.gamma = gamma
        self.v = v
        self.vae = vae

        self.optimizer = Adam(params=list(self.vae.parameters()), lr=1e-3)

    def _loss(self, xa, xb):
        xchap, mu_a, logvar_a = self.vae.forward(xa)
        mu_b, logvar_b = self.vae.encode(xb)

        std_a = torch.exp(0.5 * logvar_a)
        std_b = torch.exp(0.5 * logvar_b)

        # for those terms see Appendix: Summary of the algorithm, p.15
        # 1st term constant
        # e1 = - (xa.shape[1] / 2) * np.log(2 * np.pi * self.v) - (1 / (2 * self.v)) * torch.norm(xa - xchap,dim=1)

        # with Bernoulli decoder
        e1 = torch.sum(xa * torch.log(xchap) + (1 - xa) * torch.log(1 - xchap), dim=1)

        # 2nd term constant
        e2 = -(1 / 2) * torch.sum(torch.pow(mu_a, 2) + std_a, dim=1) - mu_a.shape[1] / 2 * np.log(2 * np.pi)

        # 2nd term constant
        e3 = -(1 / 2) * torch.sum(torch.pow(mu_b, 2) + std_b, dim=1) - mu_a.shape[1] / 2 * np.log(2 * np.pi)

        wd = entropy_regularized_wd2(mu_a, mu_b, std_a, std_b, self.gamma)

        return torch.mean(e1 + e2 + e3 - wd)

    def _adversarial_loss(self, xa, xb):
        mu_a, logvar_a = self.encode(xa)
        mu_b, logvar_b = self.encode(xb)

        std_a = torch.exp(0.5 * logvar_a)
        std_b = torch.exp(0.5 * logvar_b)

        return - wd22(mu_a, mu_b, std_a, std_b)

    def training_step(self, xa, *args, **kwargs):
        # Adversarial sample of xa
        xb = projected_gradient_descent(xa,  # Ball center
                                        lambda y: self._adversarial_loss(xa, y),  # loss function based on xa
                                        # lambda y: torch.norm(y-y), # dummy loss
                                        self.norm_fn,  # norm to use to define the ball
                                        self.ball_radius,  # radius of the ball
                                        self.alpha,  # size of step of the pgd

                                        # those two are exclusive
                                        # set steps to stop after a fixed number of iteration
                                        # set eps to stop when two successive points are eps-close for the specified
                                        # norm
                                        self.steps,  # number of steps to do
                                        self.eps,  # stopping condition

                                        device=self.device
                                        )

        loss = self._loss(xa, xb)
        self.optimizer.zero_grad()

        # Maximization step
        (-loss).backward()

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
