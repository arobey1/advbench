import torch.optim as optim
import torch

class PrimalDualOptimizer:
    def __init__(self, parameters, margin, eta):
        self.parameters = parameters
        self.margin = margin
        self.eta = eta

    def step(self, cost):
        self.parameters['dual_var'] = self.relu(self.parameters['dual_var'] + self.eta * (cost - self.margin))

    @staticmethod
    def relu(x):
        return x if x > 0 else torch.tensor(0).cuda()

