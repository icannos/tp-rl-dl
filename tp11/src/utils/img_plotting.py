from matplotlib import pyplot as plt
import numpy as np
import torch

def plt_vae_adversarial(inputs, adversarial_perturbations, outputs, p_outputs):
    n_models = len(inputs)

    fig = plt.figure(figsize=(5, n_models), dpi=100)

    for k, (input, d, output, p_output) in enumerate(zip(inputs, adversarial_perturbations, outputs, p_outputs)):

        fig.add_subplot(n_models, 5, k * 5+1)
        plt.imshow(input)
        plt.axis('off')

        fig.add_subplot(n_models, 5, k * 5+2)
        plt.imshow(output)
        plt.axis('off')

        fig.add_subplot(n_models, 5, k * 5 + 3)
        plt.imshow(d)
        plt.axis('off')

        fig.add_subplot(n_models, 5, k * 5 + 4)
        plt.imshow(input + torch.tensor(d))
        plt.axis('off')

        fig.add_subplot(n_models, 5, k * 5 + 5)
        plt.imshow(p_output)
        plt.axis('off')





