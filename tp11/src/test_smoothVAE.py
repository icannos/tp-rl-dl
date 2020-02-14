from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image

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
CUDA = False

SAVE_VAE_PATH = "../results/smooth_vae.pck1"
SAVE_CLASSIFIER_PATH = "../results/smooth_classifier.pck1"

# Hyper-parameters for VAE training
RETRAIN_VAE = True      # if False, loads trained VAE from SAVE_VAE_PATH
batch_size = 128
epochs_vae = 20
inner_dim = 200
representation_dim = 32
n_layers = 1
v = 0.25
gamma = 1
eps = 1
L = 1

# Hyper-parameters for classifier training
RETRAIN_CLASSIFIER = True      # if False, loads trained classifier from SAVE_CLASSIFIER_PATH
epochs_classifier = 10

# Hyper-parameters for adversarial accuracy evaluation
radius = 0.1
n_iterations = 100
n_restarts = 10

torch.manual_seed(SEED)
DEVICE = torch.device("CUDA" if (CUDA and torch.CUDA.is_available()) else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# torch.set_default_tensor_type('torch.cuda.FloatTensor')


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

input_dim = 784 # for MNIST
n_classes = 10


####################################################################
####################### VAE Training ###############################
####################################################################

vae = ModularVAE(input_dim,
                 [(inner_dim, nn.ReLU()) for _ in range(n_layers)] + [(representation_dim, None)],     # encoder mu shape
                 [(inner_dim, nn.ReLU()) for _ in range(n_layers)] + [(representation_dim, None)],     # encoder std shape
                 [(inner_dim, nn.ReLU()) for _ in range(n_layers)] + [(input_dim, nn.Sigmoid())])   # decoder shape

### Smooth VAE
model = SmoothVAE(vae=vae, v=v, gamma=gamma, ball_radius=eps, steps=L) #.to(DEVICE)

### Standard VAE
# model = UsualVAE(vae)

if RETRAIN_VAE or not os.path.exists(SAVE_VAE_PATH):
    for epoch in range(1, epochs_vae + 1):
        t = time.time()
        train(epoch, model, train_loader, LOG_INTERVAL, DEVICE)
        print("====> Time elapsed (seconds): {:.1f}".format(time.time()-t))
        with torch.no_grad():
            sample = torch.randn(64, representation_dim)
            sample = model.decode(sample)
            save_image(sample.view(64, 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim))),'../results/sample_vae_' + str(epoch) + '.png')


    with open(SAVE_VAE_PATH, "wb") as f:
        pickle.dump(model, f)

else:
    with open(SAVE_VAE_PATH, "rb") as f:
        model = pickle.load(f)

print(model.__class__)

####################################################################
################ Linear classifier Training ########################
####################################################################

lc = LinearClassifier(representation_dim, n_classes, model)

if RETRAIN_CLASSIFIER or not os.path.exists(SAVE_CLASSIFIER_PATH):
    for epoch in range(1, epochs_classifier + 1):
        train(epoch, lc, train_loader, LOG_INTERVAL, DEVICE, accuracy=True)

        test_acc = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data = data.to(DEVICE).view(data.shape[0], data.shape[2]*data.shape[3])
                prediction = lc(data)
                acc = torch.sum(prediction.argmax(dim=1) == target).item()
                test_acc += acc

        test_acc /= len(test_loader.dataset)
        print('====> Test set accuracy: {:.3f}'.format(test_acc))

    with open(SAVE_CLASSIFIER_PATH, "wb") as f:
        pickle.dump(lc, f)

else:
    with open(SAVE_CLASSIFIER_PATH, "rb") as f:
        lc = pickle.load(f)


####################################################################
################### Adversarial accuracy ###########################
####################################################################

print('Adversarial accuracy evaluation')

test_acc = 0
perturbed_test_acc = 0
for i, (data, target) in enumerate(test_loader):
    data = data.to(DEVICE).view(data.shape[0], data.shape[2]*data.shape[3])
    prediction = lc(data)
    acc = torch.sum(prediction.argmax(dim=1) == target).item()
    test_acc += acc

    perturbed_data = lc.projected_gradient_descent(data,
                                                      norm_fn="inf",
                                                      ball_radius=1,
                                                      alpha=0.1,
                                                      steps=n_iterations,
                                                      restarts=1,
                                                      target=target)
    perturbed_prediction = lc(perturbed_data)
    perturbed_acc = torch.sum(perturbed_prediction.argmax(dim=1) == target).item()
    perturbed_test_acc += perturbed_acc
    print('Finish batch {}, original OK: {}, perturbed OK: {}'.format(i+1, acc, perturbed_acc))

test_acc /= len(test_loader.dataset)
perturbed_test_acc /= len(test_loader.dataset)

print('====> Original test set accuracy: {:.4f}'.format(test_acc))
print('====> Perturbed test set accuracy: {:.4f}'.format(perturbed_test_acc))
