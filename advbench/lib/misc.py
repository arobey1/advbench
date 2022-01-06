import torch
import hashlib
import sys
from functools import wraps
from time import time
import pandas as pd
import torch.nn.functional as F

from advbench.lib import meters

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} took: {te-ts:.3f} sec')
        return result
    return wrap

def seed_hash(*args):
    """Derive an integer hash from all args, for use as a random seed."""

    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_full_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

@torch.no_grad()
def accuracy(algorithm, loader, device):
    correct, total = 0, 0

    algorithm.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = algorithm.predict(imgs)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.size(0)
    algorithm.train()

    return 100. * correct / total

def adv_accuracy(algorithm, loader, device, attack):
    correct, total = 0, 0

    algorithm.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = attack(imgs, labels)

        with torch.no_grad():
            output = algorithm.predict(adv_imgs)
            pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.size(0)
    algorithm.train()

    return 100. * correct / total

def sample_deltas(imgs, eps):
    return 2 * eps * torch.rand_like(imgs) - eps

def img_clamp(imgs):
    return torch.clamp(imgs, 0.0, 1.0)

@torch.no_grad()
def cvar_loss(algorithm, loader, device, test_hparams):

    beta, M = test_hparams['cvar_sgd_beta'], test_hparams['cvar_sgd_M']
    eps = test_hparams['epsilon']
    cvar_meter = meters.AverageMeter()

    algorithm.eval()
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        ts = torch.zeros(size=(imgs.size(0),)).to(device)

        for _ in range(test_hparams['cvar_sgd_n_steps']):

            cvar_loss, indicator_sum = 0, 0
            for _ in range(test_hparams['cvar_sgd_M']):
                pert_imgs = img_clamp(imgs + sample_deltas(imgs, eps))
                curr_loss = F.cross_entropy(algorithm.predict(pert_imgs), labels, reduction='none')
                indicator_sum += torch.where(curr_loss > ts, torch.ones_like(ts), torch.zeros_like(ts))
                cvar_loss += F.relu(curr_loss - ts)

            indicator_avg = indicator_sum / float(M)
            cvar_loss = (ts + cvar_loss / (M * beta)).mean()

            # gradient update on ts
            grad_ts = (1 - (1 / beta) * indicator_avg) / float(imgs.size(0))
            ts = ts - test_hparams['cvar_sgd_t_step_size'] * grad_ts

        cvar_meter.update(cvar_loss.item(), n=imgs.size(0))

    algorithm.train()

    return cvar_meter.avg

def cvar_grad_loss(algorithm, loader, device, test_hparams):

    beta, M = test_hparams['cvar_sgd_beta'], test_hparams['cvar_sgd_M']
    eps = test_hparams['epsilon']
    cvar_meter = meters.AverageMeter()
    algorithm.eval()

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        ts = torch.zeros(size=(imgs.size(0),)).to(device)

        for _ in range(test_hparams['cvar_sgd_n_steps']):
            ts.requires_grad = True
            cvar_loss = 0
            for _ in range(M):
                pert_imgs = img_clamp(imgs + sample_deltas(imgs, eps))
                curr_loss = F.cross_entropy(algorithm.predict(pert_imgs), labels, reduction='none')
                cvar_loss += F.relu(curr_loss - ts)

            cvar_loss = (ts + cvar_loss / (float(M) * beta)).mean()
            grad_ts = torch.autograd.grad(cvar_loss, [ts])[0].detach()
            ts = ts - test_hparams['cvar_sgd_t_step_size'] * grad_ts
            ts = ts.detach()

        cvar_meter.update(cvar_loss.item(), n=imgs.size(0))

    algorithm.train()

    return cvar_meter.avg


@torch.no_grad()
def augmented_accuracy(algorithm, loader, device, test_hparams):

    correct, total = 0, 0
    correct_indiv = []
    eps = test_hparams['epsilon']

    algorithm.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        batch_correct_ls = []
        for _ in range(test_hparams['aug_n_samples']):

            # sample deltas and pass perturbed images through model
            output = algorithm.predict(img_clamp(imgs + sample_deltas(imgs, eps)))
            pert_pred = output.argmax(dim=1, keepdim=True)

            # unreduced predictions
            pert_correct = pert_pred.eq(labels.view_as(pert_pred))

            batch_correct_ls.append(pert_correct)
            correct += pert_correct.sum().item()
            total += imgs.size(0)

        batch_correct = torch.sum(torch.hstack(batch_correct_ls), dim=1)
        correct_indiv.append(batch_correct)

    aug_acc = 100. * correct / total
    aug_indiv_accs = 100. * torch.hstack(correct_indiv) / test_hparams['aug_n_samples']

    def calc_beta_quant_acc(beta):
        """Calculate the quantile accuracy for the augmented samples."""
        beta_quant_indiv_accs = torch.where(
            aug_indiv_accs > (1 - beta) * 100.,
            100. * torch.ones_like(aug_indiv_accs),
            torch.zeros_like(aug_indiv_accs))
        beta_quant_acc = beta_quant_indiv_accs.mean().item()
        return beta_quant_indiv_accs, beta_quant_acc

    # loop over betas, find corresponding quantile accuracies
    beta_quant_indiv_accs, beta_quant_accs = {}, {}
    for beta in test_hparams['test_betas']:
        quant_indiv_acc, quant_acc = calc_beta_quant_acc(beta)
        beta_quant_indiv_accs[beta] = quant_indiv_acc
        beta_quant_accs[beta] = quant_acc

    return aug_acc, aug_indiv_accs, beta_quant_indiv_accs, beta_quant_accs


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()