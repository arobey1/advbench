import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from advbench import networks
from advbench import optimizers
from advbench import attacks
from advbench.lib import meters

ALGORITHMS = [
    'ERM',
    'PGD',
    'FGSM',
    'TRADES',
    'ALP',
    'CLP',
    'Gaussian_DALE',
    'Laplacian_DALE',
    'Gaussian_DALE_PD'
]

class Algorithm(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.classifier = networks.Classifier(
            input_shape, num_classes, hparams)
        self.optimizer = optimizers.Optimizer(
            self.classifier, hparams)
        self.device = device
        
        self.meters = OrderedDict()
        self.meters['loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        raise NotImplementedError

    def predict(self, imgs):
        return self.classifier(imgs)

class ERM(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ERM, self).__init__(input_shape, num_classes, hparams, device)

    def step(self, imgs, labels):
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(imgs), labels)
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

class PGD(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(PGD, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)

    def step(self, imgs, labels):

        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(adv_imgs), labels)
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}        

class FGSM(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(FGSM, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.FGSM_Linf(self.classifier, self.hparams, device)

    def step(self, imgs, labels):

        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(adv_imgs), labels)
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

class TRADES(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(TRADES, self).__init__(input_shape, num_classes, hparams, device)
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')  # TODO(AR): let's write a method to do the log-softmax part
        self.attack = attacks.TRADES_Linf(self.classifier, self.hparams, device)
        
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['invariance loss'] = meters.AverageMeter()

    def step(self, imgs, labels):

        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        robust_loss = self.kl_loss_fn(
            F.log_softmax(self.predict(adv_imgs), dim=1),
            F.softmax(self.predict(imgs), dim=1))
        total_loss = clean_loss + self.hparams['trades_beta'] * robust_loss
        total_loss.backward()
        self.optimizer.step()

        return {'loss': total_loss.item()}

class LogitPairingBase(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(LogitPairingBase, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)

    def pairing_loss(self, imgs, adv_imgs):
        logit_diff = self.predict(adv_imgs) - self.predict(imgs)
        return torch.norm(logit_diff, dim=1).mean()

class ALP(LogitPairingBase):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ALP, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)
        
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
    def __init__(self, input_shape, num_classes, hparams, device):
        super(CLP, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)

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
    def __init__(self, input_shape, num_classes, hparams, device):
        super(MART, self).__init__(input_shape, num_classes, hparams, device)
        self.kl_loss_fn = nn.KLDivLoss(reduction='none')
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)

    def step(self, imgs, labels):
        
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_output = self.classifier(imgs)
        adv_output = self.classifier(adv_imgs)
        adv_probs = F.softmax(adv_output, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_label = torch.where(tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(adv_output, labels) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_label)
        nat_probs = F.softmax(clean_output, dim=1)
        true_probs = torch.gather(nat_probs, 1, (labels.unsqueeze(1)).long()).squeeze()
        loss_robust = (1.0 / imgs.size(0)) * torch.sum(
            torch.sum(self.kl_loss_fn(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss = loss_adv + self.hparams['mart_beta'] * loss_robust
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MMA(Algorithm):
    pass

# with and without primal dual

class Gaussian_DALE(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Gaussian_DALE, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.LMC_Gaussian_Linf(self.classifier, self.hparams, device)
        
    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        total_loss = robust_loss + self.hparams['g_dale_nu'] * clean_loss
        total_loss.backward()
        self.optimizer.step()

        return {'loss': total_loss.item()}

class Laplacian_DALE(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Laplacian_DALE, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.LMC_Laplacian_Linf(self.classifier, self.hparams, device)

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        total_loss = robust_loss + self.hparams['l_dale_nu'] * clean_loss
        total_loss.backward()
        self.optimizer.step()

        return {'loss': total_loss.item()}

class PrimalDualBase(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(PrimalDualBase, self).__init__(input_shape, num_classes, hparams, device)
        self.dual_params = {'dual_var': torch.tensor(1.0).to(self.device)}

class Gaussian_DALE_PD(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Gaussian_DALE_PD, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.LMC_Gaussian_Linf(self.classifier, self.hparams, device)
        self.pd_optimizer = optimizers.PrimalDualOptimizer(
            parameters=self.dual_params,
            margin=self.hparams['g_dale_pd_margin'],
            eta=self.hparams['g_dale_pd_step_size'])

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        total_loss = robust_loss + self.dual_params['dual_var'] * clean_loss
        total_loss.backward()
        self.optimizer.step()
        self.pd_optimizer.step(clean_loss.detach())

        return {'loss': total_loss.item(), 'dual_var': self.dual_params['dual_var']}
