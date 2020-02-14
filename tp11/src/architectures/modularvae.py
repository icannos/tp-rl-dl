import torch
import torch.utils.data
from torch import nn


class ModularVAE(nn.Module):
    def __init__(self,
                 input_dim,
                 encoder_mu_shape,
                 encoder_std_shape,
                 decoder_shape
                 ):
        """
        Build a VAE with the specified shape.
        :param input_dim: Dimension of an input vector
        :param encoder_mu_shape: List of tuple. [ (dim, activation_function), ... ]
        :param encoder_std_shape: List of tuple. [ (dim, activation_function), ... ]
        :param decoder_shape: List of tuple. [ (dim, activation_function) ... ]
        """
        super(ModularVAE, self).__init__()

        if encoder_mu_shape[-1][0] != encoder_std_shape[-1][0]:
            raise Exception("The output of the mean estimator and the output of the std estimator should"
                            " have the save dimension.")

        # if encoder_mu_shape[-1][0] != decoder_shape[0][0]:
        #     raise Exception("The dimension of the encoded vector is not the same as the dimension of the decoder.")

        self.decoder_shape = decoder_shape
        self.encoder_std_shape = encoder_std_shape
        self.encoder_mu_shape = encoder_mu_shape
        self.input_dim = input_dim

        self.encoder_mu = nn.Sequential(*self._mk_layers(input_dim, encoder_mu_shape))
        self.encoder_std = nn.Sequential(*self._mk_layers(input_dim, encoder_std_shape))

        self.decoder = nn.Sequential(*self._mk_layers(encoder_mu_shape[-1][0], decoder_shape))

    @staticmethod
    def _mk_layers(input_dim, shape):
        n = len(shape)

        layers = [nn.Linear(input_dim, shape[0][0]), shape[0][1]]

        for i in range(1, n):
            s, activation = shape[i]
            prev_s, _ = shape[i - 1]
            layers.append(nn.Linear(prev_s, s))
            if activation is not None:
                layers.append(activation)

        return layers

    def encode(self, x):
        mu = self.encoder_mu(x)
        logvar = self.encoder_std(x)

        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
