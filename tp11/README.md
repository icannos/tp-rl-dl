# Robust VAE

Reproduction of the paper "Adversarially Robust Representations with Smooth Encoders" and basic variationnal autoencoder, currently under review at ICLR 2020: https://openreview.net/forum?id=H1gfFaEYDS

Initial implementations of VAE and Projected GD attack are based respectively on the [pytorch documentation](https://github.com/pytorch/examples/blob/master/vae/main.py) and on this [github gist](https://gist.github.com/oscarknagg/45b187c236c6262b1c4bbe2d0920ded6).

## Usage

## Experiments

## Project Architecture
The `data` directory will be create at runtime the first time you launch the experiments and the data will be downloaded in it.

We took car to split our programm in order to make it reusable. The different `VAE` models are available in `src/architectures` and the wrappers to train it with the usual method (resp. the method from the paper) is available in `src/usualVAE.py` (resp. in `src/smoothVAE.py`).

We also have a hand-made implementation of the projected gradient descent (PGD) algorithm in `src/utils/projected_gradient_descent.py`.

```
.
├── data
│   └── MNIST			# Our report
└── src				# Contains the sources 
    ├── architectures		# The different models
    └── utils			# Some useful tools such as pgd
```


