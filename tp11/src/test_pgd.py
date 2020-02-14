import numpy as np
from utils.projected_gradient_descent import projected_gradient_descent
import torch


def test(a,b):
    return torch.norm(b - torch.Tensor([[0.5,0.5]]), dim=1)

if __name__ == "__main__":
    print("Test:")
    torch.manual_seed(42)

    x = torch.Tensor([[1,2], [2., 3.], [1, 2]])
    # x = torch.Tensor([[1,2]])
    n_x = projected_gradient_descent(x, lambda y: test(x, y), torch.norm, 3, steps=1000)

    print(torch.norm(x - n_x, dim=1))
    print(test(x, n_x))
    print(x)
    print(n_x)
