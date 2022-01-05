import torch
import hashlib
import sys
from functools import wraps
from time import time
import pandas as pd

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

def augmented_accuracy(algorithm, loader, device, betas, eps, n_samples):

    def img_clamp(imgs):
        return torch.clamp(imgs, 0.0, 1.0)

    correct, total = 0, 0
    correct_indiv = []

    algorithm.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        batch_correct_ls = []
        for _ in range(n_samples):
            with torch.no_grad():

                # sample deltas and pass perturbed images through model
                samp_deltas = 2 * eps * torch.rand_like(imgs) - eps
                output = algorithm.predict(img_clamp(imgs + samp_deltas))
                pert_pred = output.argmax(dim=1, keepdim=True)

                # unreduced predictions
                pert_correct = pert_pred.eq(labels.view_as(pert_pred))

                batch_correct_ls.append(pert_correct)
                correct += pert_correct.sum().item()
                total += imgs.size(0)

        batch_correct = torch.sum(torch.hstack(batch_correct_ls), dim=1)
        correct_indiv.append(batch_correct)

    aug_acc = 100. * correct / total
    aug_indiv_accs = 100. * torch.hstack(correct_indiv) / n_samples

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
    for beta in betas:
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