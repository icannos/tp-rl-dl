import torch

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

from torch.optim import Adam
import torch.utils.data
from torch import nn
from torchvision import datasets, transforms
import numpy as np
import pickle
from architectures.modularvae import ModularVAE
from usualVAE import UsualVAE
from utils.linear_classifier import LinearClassifier
from utils.img_plotting import plt_vae_adversarial
import matplotlib.pyplot as plt

SAVE_CLASSIFIER_PATH = "../results/smooth_classifier.pck1"

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=16, shuffle=True)


print("test")
with open(SAVE_CLASSIFIER_PATH, 'rb') as f:
    lc = pickle.load(f).to('cpu')

print(lc)

def adversarial_attack(model, input, target_class, l=1):
    d = torch.rand_like(input, requires_grad=True)

    optimizer = Adam(params=[d])

    for i in range(1000):
        pred = torch.nn.functional.softmax(model(input + d), dim=1)
        print(pred)
        loss = l * torch.norm(d) + nn.functional.mse_loss(pred, target_class, reduction="mean")
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return d.cpu().detach().numpy()


def vae_attack(model, input, target_class):
    print(target_class)

    output, _, _ = model.vae(torch.Tensor(input))

    output = output.cpu().detach().numpy()

    d = adversarial_attack(model, input, target_class)

    p_output, _, _ = model.vae(torch.Tensor(input + torch.tensor([d])))
    p_output = p_output.cpu().detach().numpy()

    return [input, d, output, p_output]


input, y = test_loader.dataset[0]
input = np.reshape(input, (1, 784))

target_class = np.zeros((1,10))
target_class[0][y] = 0

input, d, output, p_output = vae_attack(lc, input, torch.Tensor(target_class))

plt_vae_adversarial([np.reshape(input, (28,28))], [np.reshape(d, (28,28))], [np.reshape(output, (28,28))], [np.reshape(p_output, (28,28))])

plt.savefig("../results/usual_attack.png")
