import torch
import numpy as np


def proj(x, center, norm_fn, eps):
    """
    Project x on the ball centered in center with radius eps. Unit ball: center = 0, eps = 1
    :param x:
    :param center:
    :param norm_fn: norm to use to define the ball
    :param eps:
    :return: proj on ball
    """
    if norm_fn == "inf": 
        norm = torch.max(torch.abs(x - center), dim=1)[0].view(-1, 1)
    else: 
        norm = torch.norm(x-center, dim=1).view(-1,1)
    return (center + eps * (x-center) / norm) * (norm > eps).float() + x * (1 - (norm > eps).float())


def pgd_step(xa, xb, loss_fn, norm_fn, ball_radius, alpha):
    loss = loss_fn(xb)

    # computes the loss gradient
    loss.sum().backward()

    # we dont want to store the gradient of this operation
    with torch.no_grad():
        
        if norm_fn=="inf":
                gradient = alpha * xb.grad.sign()
        else:
                gradient = alpha * xb.grad / torch.norm(xb.grad, dim=1).view(-1,1)
    
        
        # This makes exactly this: projB ( xb - alpha * grad_f (xb) )
        # It is the fixpoint iteration
        # More info here: https://tlienart.github.io/pub/csml/cvxopt/pgd.html
        xb = proj(xb - gradient, xa, norm_fn, ball_radius)

    return xb, loss_fn(xb)


def projected_gradient_descent(xa, loss_fn, norm_fn, ball_radius, alpha=0.01, steps=None, eps=None, restarts=0, device='cpu'):
    """

    :param xa: ball center
    :param loss_fn: objective to maximize inside the ball
    :param norm_fn: norm to use to define the ball
    :param ball_radius: radius of the ball
    :param alpha: pgd step size
    :param steps: Number of steps of descent to do
    :param eps: distance between two iteration to stop
    :param restarts: number of authorized restarts
    :return:
    """

    if steps and eps:
        raise Exception("You have to choose between a fix number of steps and an eps as stop condition.")

    if steps is None and eps is None:
        steps = 3 # If nothing is specified we do 3 steps. This is totally arbitrary. I decided it.

    best_loss = torch.Tensor([np.inf]*xa.shape[0]).to(device)
    final_xb = torch.zeros_like(xa).to(device)

    for _ in range(restarts+1):
        if steps:
            # Make a copy and get rid of the old gradient, some dark magic stolen on the internet
            xb = (xa + torch.randn_like(xa)).clone().detach().requires_grad_(True).to(device)

            for _ in range(steps):
                xb = xb.clone().detach().requires_grad_(True).to(device)
                # make a fixpoint iteration
                xb, loss = pgd_step(xa, xb, loss_fn, norm_fn, ball_radius, alpha)

        elif eps:
            xb = xa
            # Make a copy and get rid of the old gradient, some dark magic stolen on the internet
            next_xb = (xa + torch.randn_like(xa)).clone().detach().requires_grad_(True).to(device)

            while norm_fn(xb - next_xb) > eps:
                xb = next_xb.clone().detach().requires_grad_(True).to(device)
                # fixpoint iteration
                next_xb, loss = pgd_step(xa, xb, loss_fn, norm_fn, ball_radius, alpha)

        final_xb[torch.where(loss < best_loss)[0]] = xb[torch.where(loss < best_loss)[0]]
        best_loss = torch.min(loss, best_loss)

    return final_xb


# Just some tests, you can skip it

def test(a, b):
    return torch.exp(torch.norm(b + 3*a))

if __name__ == "__main__":
    print("Test:")
    torch.manual_seed(42)

    t = torch.Tensor([[1, 2], [2., 3.], [1, 2]])
    # t = torch.Tensor([[1,2]])
    print(projected_gradient_descent(t, lambda y: test(t, y), torch.norm, 1, eps=0.001, restarts=2))
