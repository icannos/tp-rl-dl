import torch
from torch import nn
from torch.optim import Adam

from utils.projected_gradient_descent import projected_gradient_descent

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, vae):
        super(LinearClassifier, self).__init__()
        
        self.vae = vae
        self.classifier = torch.nn.Linear(input_dim, output_dim)

        self.optimizer = Adam(params=list(self.classifier.parameters()))
        self.loss = nn.CrossEntropyLoss(reduction="mean")
    
    def training_step(self, data, target):
        l = self.loss(self.classifier(self.vae.encode(data)[0]), target)	# we keep only the means

        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return l

    def __call__(self, x):
        return self.classifier(self.vae.encode(x)[0])


    def projected_gradient_descent(self, xa, norm_fn, ball_radius, alpha, steps, restarts=0, target=None):
        # Adversarial sample of xa
        target_onehot = torch.FloatTensor(target.shape[0], 10)
        target_onehot.zero_()
        target_onehot.scatter_(1, target.view(-1, 1), 1)

        return projected_gradient_descent(xa, # Ball center
                                        #   lambda y: torch.norm(y-y), # dummy loss
                                        #   lambda y: nn.CrossEntropyLoss()(self(y), target), # cross entropy loss
                                          lambda y: self(y).gather(1, target.view(-1, 1)).sum(),  # value at true label
                                        #   lambda y: torch.norm(nn.Softmax(dim=1)(self(y)) - target_onehot),  # MSE loss
                                          norm_fn, # norm to use to define the ball
                                          ball_radius, # radius of the ball
                                          alpha, # size of step of the pgd

                                          # those two are exclusive
                                          # set steps to stop after a fixed number of iteration
                                          # set eps to stop when two successive points are eps-close for the specified
                                          # norm
                                          steps, # number of steps to do
                                          None, # stopping condition

                                          restarts    # number of restarts allowed
                                          )
