import torch.optim as optim
import torch

#TODO(AR): Need to write an optimizer for primal-dual

def Optimizer(classifier, hparams):

    # return optim.Adadelta(
    #     classifier.parameters(),
    #     lr=1.0)

    return optim.SGD(
        classifier.parameters(),
        lr=hparams['learning_rate'],
        momentum=hparams['sgd_momentum'],
        weight_decay=hparams['weight_decay'])

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

