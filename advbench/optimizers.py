import torch.optim as optim

#TODO(AR): Need to write an optimizer for primal-dual

def Optimizer(classifier, hparams):

    return optim.Adadelta(
        classifier.parameters(),
        lr=1.0)

    return optim.SGD(
        classifier.parameters(),
        lr=hparams['learning_rate'],
        momentum=hparams['sgd_momentum'],
        weight_decay=hparams['weight_decay'])