import torch
import torch.nn as nn
import torch.nn.functional as F

from advbench import networks
from advbench import optimizers
from advbench import attacks

class Algorithm(nn.Module):
    def __init__(self, input_shape, num_classes, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.classifier = networks.Classifier(
            input_shape, num_classes, hparams)
        self.optimizer = optimizers.Optimizer(
            self.classifier, hparams)

    def step(self, imgs, labels):
        raise NotImplementedError

    def predict(self, imgs):
        return self.classifier(imgs)

class ERM(Algorithm):
    def __init__(self, input_shape, num_classes, hparams):
        super(ERM, self).__init__(input_shape, num_classes, hparams)

    def step(self, imgs, labels):
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(imgs), labels)
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

class PGD(Algorithm):
    def __init__(self, input_shape, num_classes, hparams):
        super(PGD, self).__init__(input_shape, num_classes, hparams)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams)

    def step(self, imgs, labels):

        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(adv_imgs), labels)
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

class TRADES(Algorithm):
    def __init__(self, input_shape, num_classes, hparams):
        super(TRADES, self).__init__(input_shape, num_classes, hparams)
        self.kl_loss_fn = nn.KLDivLoss(size_average=False)  # AR: let's write a method to do the log-softmax part
        self.attack = attacks.TRADES_Linf(self.classifier, self.hparams)

    def step(self, imgs, labels):

        batch_size = imgs.size(0)
        adv_imgs = self._trades_iteration_linf(imgs)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        robust_loss = (1. / batch_size) * self.kl_loss_fn(
            F.log_softmax(self.predict(adv_imgs), dim=1),
            F.softmax(self.predict(imgs), dim=1))
        total_loss = clean_loss + self.hparams['trades_beta'] * robust_loss
        total_loss.backward()
        self.optimizer.step()

        return {'loss': total_loss.item()}

class LogitPairingBase(Algorithm):
    def __init__(self, input_shape, num_classes, hparams):
        super(LogitPairingBase, self).__init__(input_shape, num_classes, hparams)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams)

    def pairing_loss(self, imgs, adv_imgs):
        logit_diff = self.predict(adv_imgs) - self.predict(imgs)
        return torch.norm(logit_diff, dim=1).mean()

class ALP(LogitPairingBase):
    def __init__(self, input_shape, num_classes, hparams):
        super(ALP, self).__init__(input_shape, num_classes, hparams)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams)
        
    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        logit_pairing_loss = self.pairing_loss(imgs, adv_imgs)
        total_loss = robust_loss + logit_pairing_loss
        total_loss.backward()
        self.optimizer.step()

        return {'loss': total_loss.item()}

class CLP(LogitPairingBase):
    def __init__(self, input_shape, num_classes, hparams):
        super(CLP, self).__init__(input_shape, num_classes, hparams)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams)

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        robust_loss = F.cross_entropy(self.predict(imgs), labels)
        logit_pairing_loss = self.pairing_loss(imgs, adv_imgs)
        total_loss = robust_loss + logit_pairing_loss
        total_loss.backward()
        self.optimizer.step()

        return {'loss': total_loss.item()}

class MART(Algorithm):
    def __init__(self, input_shape, num_classes, hparams):
        super(MART, self).__init__(input_shape, num_classes, hparams)
        self.kl_loss_fn = nn.KLDivLoss(size_average=False)

    def step(self, imgs, labels):
        pass


class MMA(Algorithm):
    pass

# with and without primal dual

class DALE_GaussianHMC(Algorithm):
    pass

class DALE_LaplacianHMC(Algorithm):
    pass

