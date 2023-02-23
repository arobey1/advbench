import torch
import hashlib
import sys
import os
import json
from functools import wraps
from time import time
import pandas as pd
import torch.nn.functional as F
import numpy as np

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

def print_row(row, col_width=10):
    sep, end_ = "  ", ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = f'{x:.5f}'
        return str(x).ljust(col_width)[:col_width]
    
    print(sep.join([format_val(x) for x in row]), end_)

def stage_path(data_dir, name):
    path = os.path.join(data_dir, name)
    os.makedirs(path) if not os.path.exists(path) else None
    return path

def read_dict(fname):
    with open(fname, 'r') as f:
        d = json.load(f)
    return d

def print_full_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

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