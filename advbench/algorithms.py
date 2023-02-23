import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch.optim as optim

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
    'Gaussian_DALE_PD',
    'Gaussian_DALE_PD_Reverse',
    'KL_DALE_PD',
    'CVaR_SGD',
    'CVaR_SGD_Autograd',
    'CVaR_SGD_PD',
    'ERM_DataAug',
    'TERM',
    'RandSmoothing'
]

class Algorithm(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.classifier = networks.Classifier(
            input_shape, num_classes, hparams)
        self.optimizer = optim.SGD(
            self.classifier.parameters(),
            lr=hparams['learning_rate'],
            momentum=hparams['sgd_momentum'],
            weight_decay=hparams['weight_decay'])
        self.device = device
        
        self.meters = OrderedDict()
        self.meters['Loss'] = meters.AverageMeter()
        self.meters_df = None

    def step(self, imgs, labels):
        raise NotImplementedError

    def predict(self, imgs):
        return self.classifier(imgs)

    @staticmethod
    def img_clamp(imgs):
        return torch.clamp(imgs, 0.0, 1.0)

    def reset_meters(self):
        for meter in self.meters.values():
            meter.reset()

    def meters_to_df(self, epoch):
        if self.meters_df is None:
            columns = ['Epoch'] + list(self.meters.keys())
            self.meters_df = pd.DataFrame(columns=columns)

        values = [epoch] + [m.avg for m in self.meters.values()]
        self.meters_df.loc[len(self.meters_df)] = values
        return self.meters_df

class ERM(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ERM, self).__init__(input_shape, num_classes, hparams, device)

    def step(self, imgs, labels):
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(imgs), labels)
        loss.backward()
        self.optimizer.step()
        
        self.meters['Loss'].update(loss.item(), n=imgs.size(0))

class ERM_DataAug(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ERM_DataAug, self).__init__(input_shape, num_classes, hparams, device)

    def sample_deltas(self, imgs):
        eps = self.hparams['epsilon']
        return 2 * eps * torch.rand_like(imgs) - eps

    def step(self, imgs, labels):
        self.optimizer.zero_grad()
        loss = 0
        for _ in range(self.hparams['cvar_sgd_M']):
            loss += F.cross_entropy(self.predict(imgs), labels)

        loss = loss / float(self.hparams['cvar_sgd_M'])
        loss.backward()
        self.optimizer.step()
        
        self.meters['Loss'].update(loss.item(), n=imgs.size(0))

class TERM(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(TERM, self).__init__(input_shape, num_classes, hparams, device)
        self.meters['tilted loss'] = meters.AverageMeter()
        self.t = torch.tensor(self.hparams['term_t'])

    def step(self, imgs, labels):
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(imgs), labels, reduction='none')
        term_loss = torch.log(torch.exp(self.t * loss).mean() + 1e-6) / self.t
        term_loss.backward()
        self.optimizer.step()
        
        self.meters['Loss'].update(loss.mean().item(), n=imgs.size(0))
        self.meters['tilted loss'].update(term_loss.item(), n=imgs.size(0))

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

        self.meters['Loss'].update(loss.item(), n=imgs.size(0))

class RandSmoothing(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(RandSmoothing, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.SmoothAdv(self.classifier, self.hparams, device)

    def step(self, imgs, labels):

        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()        
        loss = F.cross_entropy(self.predict(adv_imgs), labels)
        loss.backward()
        self.optimizer.step()

        self.meters['Loss'].update(loss.item(), n=imgs.size(0))

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

        self.meters['Loss'].update(loss.item(), n=imgs.size(0))

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

        self.meters['Loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['invariance loss'].update(robust_loss.item(), n=imgs.size(0))

        return {'loss': total_loss.item()}

class LogitPairingBase(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(LogitPairingBase, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)
        self.meters['logit loss'] = meters.AverageMeter()

    def pairing_loss(self, imgs, adv_imgs):
        logit_diff = self.predict(adv_imgs) - self.predict(imgs)
        return torch.norm(logit_diff, dim=1).mean()

class ALP(LogitPairingBase):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ALP, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)
        self.meters['robust loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        logit_pairing_loss = self.pairing_loss(imgs, adv_imgs)
        total_loss = robust_loss + logit_pairing_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['Loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['logit loss'].update(logit_pairing_loss.item(), n=imgs.size(0))

class CLP(LogitPairingBase):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(CLP, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)

        self.meters['clean loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        logit_pairing_loss = self.pairing_loss(imgs, adv_imgs)
        total_loss = clean_loss + logit_pairing_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['Loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['logit loss'].update(logit_pairing_loss.item(), n=imgs.size(0))

class MART(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(MART, self).__init__(input_shape, num_classes, hparams, device)
        self.kl_loss_fn = nn.KLDivLoss(reduction='none')
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)

        self.meters['robust loss'] = meters.AverageMeter()
        self.meters['invariance loss'] = meters.AverageMeter()

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

        self.meters['Loss'].update(loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(loss_robust.item(), n=imgs.size(0))
        self.meters['invariance loss'].update(loss_adv.item(), n=imgs.size(0))


class MMA(Algorithm):
    pass

class Gaussian_DALE(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Gaussian_DALE, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.LMC_Gaussian_Linf(self.classifier, self.hparams, device)
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['robust loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        total_loss = robust_loss + self.hparams['g_dale_nu'] * clean_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['Loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))

class Laplacian_DALE(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Laplacian_DALE, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.LMC_Laplacian_Linf(self.classifier, self.hparams, device)
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['robust loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        total_loss = robust_loss + self.hparams['l_dale_nu'] * clean_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['Loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))

class PrimalDualBase(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(PrimalDualBase, self).__init__(input_shape, num_classes, hparams, device)
        self.dual_params = {'dual_var': torch.tensor(1.0).to(self.device)}
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['robust loss'] = meters.AverageMeter()
        self.meters['dual variable'] = meters.AverageMeter()

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

        self.meters['Loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].item(), n=1)

class CVaR_SGD_Autograd(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(CVaR_SGD_Autograd, self).__init__(input_shape, num_classes, hparams, device)
        self.meters['avg t'] = meters.AverageMeter()
        self.meters['plain loss'] = meters.AverageMeter()

    def sample_deltas(self, imgs):
        eps = self.hparams['epsilon']
        return 2 * eps * torch.rand_like(imgs) - eps

    def step(self, imgs, labels):

        beta, M = self.hparams['cvar_sgd_beta'], self.hparams['cvar_sgd_M']
        ts = torch.ones(size=(imgs.size(0),)).to(self.device)

        self.optimizer.zero_grad()
        for _ in range(self.hparams['cvar_sgd_n_steps']):

            ts.requires_grad = True
            cvar_loss = 0
            for _ in range(M):
                pert_imgs = self.img_clamp(imgs + self.sample_deltas(imgs))
                curr_loss = F.cross_entropy(self.predict(pert_imgs), labels, reduction='none')
                cvar_loss += F.relu(curr_loss - ts)
    
            cvar_loss = (ts + cvar_loss / (float(M) * beta)).mean()
            grad_ts = torch.autograd.grad(cvar_loss, [ts])[0].detach()
            ts = ts - self.hparams['cvar_sgd_t_step_size'] * grad_ts
            ts = ts.detach()

        plain_loss, cvar_loss = 0, 0
        for _ in range(M):
            pert_imgs = self.img_clamp(imgs + self.sample_deltas(imgs))
            curr_loss = F.cross_entropy(self.predict(pert_imgs), labels, reduction='none')
            plain_loss += curr_loss.mean()
            cvar_loss += F.relu(curr_loss - ts)

        cvar_loss = (cvar_loss / (beta * float(M))).mean()   

        cvar_loss.backward()
        self.optimizer.step()

        self.meters['Loss'].update(cvar_loss.item(), n=imgs.size(0))
        self.meters['avg t'].update(ts.mean().item(), n=imgs.size(0))
        self.meters['plain loss'].update(plain_loss.item() / M, n=imgs.size(0))

class CVaR_SGD(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(CVaR_SGD, self).__init__(input_shape, num_classes, hparams, device)
        self.meters['avg t'] = meters.AverageMeter()
        self.meters['plain loss'] = meters.AverageMeter()

    def sample_deltas(self, imgs):
        eps = self.hparams['epsilon']
        return 2 * eps * torch.rand_like(imgs) - eps

    def step(self, imgs, labels):

        beta = self.hparams['cvar_sgd_beta']
        M = self.hparams['cvar_sgd_M']
        ts = torch.ones(size=(imgs.size(0),)).to(self.device)

        self.optimizer.zero_grad()
        for _ in range(self.hparams['cvar_sgd_n_steps']):

            plain_loss, cvar_loss, indicator_sum = 0, 0, 0
            for _ in range(self.hparams['cvar_sgd_M']):
                pert_imgs = self.img_clamp(imgs + self.sample_deltas(imgs))
                curr_loss = F.cross_entropy(self.predict(pert_imgs), labels, reduction='none')
                indicator_sum += torch.where(curr_loss > ts, torch.ones_like(ts), torch.zeros_like(ts))

                plain_loss += curr_loss.mean()
                cvar_loss += F.relu(curr_loss - ts)                

            indicator_avg = indicator_sum / float(M)
            cvar_loss = (ts + cvar_loss / (float(M) * beta)).mean()

            # gradient update on ts
            grad_ts = (1 - (1 / beta) * indicator_avg) / float(imgs.size(0))
            ts = ts - self.hparams['cvar_sgd_t_step_size'] * grad_ts

        cvar_loss.backward()
        self.optimizer.step()

        self.meters['Loss'].update(cvar_loss.item(), n=imgs.size(0))
        self.meters['avg t'].update(ts.mean().item(), n=imgs.size(0))
        self.meters['plain loss'].update(plain_loss.item() / M, n=imgs.size(0))

class CVaR_SGD_PD(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(CVaR_SGD_PD, self).__init__(input_shape, num_classes, hparams, device)
        self.dual_params = {'dual_var': torch.tensor(1.0).to(self.device)}
        self.meters['avg t'] = meters.AverageMeter()
        self.meters['plain loss'] = meters.AverageMeter()
        self.meters['dual variable'] = meters.AverageMeter()
        self.pd_optimizer = optimizers.PrimalDualOptimizer(
            parameters=self.dual_params,
            margin=self.hparams['g_dale_pd_margin'],
            eta=self.hparams['g_dale_pd_step_size'])

    def sample_deltas(self, imgs):
        eps = self.hparams['epsilon']
        return 2 * eps * torch.rand_like(imgs) - eps

    def step(self, imgs, labels):

        beta = self.hparams['cvar_sgd_beta']
        M = self.hparams['cvar_sgd_M']
        ts = torch.ones(size=(imgs.size(0),)).to(self.device)

        self.optimizer.zero_grad()
        for _ in range(self.hparams['cvar_sgd_n_steps']):

            plain_loss, cvar_loss, indicator_sum = 0, 0, 0
            for _ in range(self.hparams['cvar_sgd_M']):
                pert_imgs = self.img_clamp(imgs + self.sample_deltas(imgs))
                curr_loss = F.cross_entropy(self.predict(pert_imgs), labels, reduction='none')
                indicator_sum += torch.where(curr_loss > ts, torch.ones_like(ts), torch.zeros_like(ts))

                plain_loss += curr_loss.mean()
                cvar_loss += F.relu(curr_loss - ts)                

            indicator_avg = indicator_sum / float(M)
            cvar_loss = (ts + cvar_loss / (float(M) * beta)).mean()

            # gradient update on ts
            grad_ts = (1 - (1 / beta) * indicator_avg) / float(imgs.size(0))
            ts = ts - self.hparams['cvar_sgd_t_step_size'] * grad_ts

        loss = cvar_loss + self.dual_params['dual_var'] * (plain_loss / float(M))
        loss.backward()
        self.optimizer.step()
        self.pd_optimizer.step(plain_loss.detach() / M)

        self.meters['Loss'].update(cvar_loss.item(), n=imgs.size(0))
        self.meters['avg t'].update(ts.mean().item(), n=imgs.size(0))
        self.meters['plain loss'].update(plain_loss.item() / M, n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].item(), n=1)

class Gaussian_DALE_PD_Reverse(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Gaussian_DALE_PD_Reverse, self).__init__(input_shape, num_classes, hparams, device)
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
        total_loss = clean_loss + self.dual_params['dual_var'] * robust_loss
        total_loss.backward()
        self.optimizer.step()
        self.pd_optimizer.step(robust_loss.detach())

        self.meters['Loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].item(), n=1)

class KL_DALE_PD(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(KL_DALE_PD, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.TRADES_Linf(self.classifier, self.hparams, device)
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.pd_optimizer = optimizers.PrimalDualOptimizer(
            parameters=self.dual_params,
            margin=self.hparams['g_dale_pd_margin'],
            eta=self.hparams['g_dale_pd_step_size'])

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        robust_loss = self.kl_loss_fn(
            F.log_softmax(self.predict(adv_imgs), dim=1),
            F.softmax(self.predict(imgs), dim=1))
        total_loss = robust_loss + self.dual_params['dual_var'] * clean_loss
        total_loss.backward()
        self.optimizer.step()
        self.pd_optimizer.step(clean_loss.detach())

        self.meters['Loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].item(), n=1)