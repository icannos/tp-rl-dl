from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image

from architectures.mnistvae import StaticMNISTVAE

import numpy as np
import pickle
import os
import time

# Architecture to use
from architectures.modularvae import ModularVAE

# Training methods

from smoothVAE import SmoothVAE
from usualVAE import UsualVAE

from utils.training_loop import train
from utils.linear_classifier import LinearClassifier

####################################################################
####################### Parameters #################################
####################################################################

# Technical parameters
SEED = 1
LOG_INTERVAL = 50
CUDA = True
SAVE_VAE_PATH = "../tmp/vae.pck1"
SAVE_CLASSIFIER_PATH = "../tmp/smooth_vae.pck1"

# Hyper-parameters for VAE training
RETRAIN_VAE = False  # if False, loads trained VAE from SAVE_VAE_PATH
batch_size = 128
epochs_vae = 100
inner_dim = 400
representation_dim = 20
n_layers = 1
v = 0.25
gamma = 1
eps = 0.1
L = 20

# Hyper-parameters for classifier training
RETRAIN_CLASSIFIER = True  # if False, loads trained classifier from SAVE_CLASSIFIER_PATH
epochs_classifier = 5

# Hyper-parameters for adversarial accuracy evaluation
radius = 0.1
n_iterations = 100
n_restarts = 10

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if (CUDA and torch.cuda.is_available()) else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}


####################################################################
####################### Data loading ###############################
####################################################################

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

input_dim = 784  # for MNIST
n_classes = 10

####################################################################
####################### VAE Training ###############################
####################################################################

# vae = ModularVAE(input_dim,
#                  [(inner_dim, nn.ReLU()) for _ in range(n_layers)] + [(representation_dim, None)],  # encoder mu shape
#                  [(inner_dim, nn.ReLU()) for _ in range(n_layers)] + [(representation_dim, None)],  # encoder std shape
#                  [(inner_dim, nn.ReLU()) for _ in range(n_layers)] + [(input_dim, nn.Sigmoid())])  # decoder shape

vae = StaticMNISTVAE().to(DEVICE)

### Smooth VAE
model = SmoothVAE(vae=vae, v=v, gamma=gamma, ball_radius=eps, steps=L, device=DEVICE)

### Standard VAE
#model = UsualVAE(vae)

for e in range(epochs_vae):
    train(e, model, train_loader, LOG_INTERVAL, DEVICE)

####################################################################
################ Linear classifier Training ########################
####################################################################

lc = LinearClassifier(representation_dim, n_classes, model).to(DEVICE)

for e in range(epochs_classifier):
    train(e, lc, train_loader, LOG_INTERVAL, DEVICE)


with open(SAVE_CLASSIFIER_PATH, 'wb') as f:
    pickle.dump(lc, f)
