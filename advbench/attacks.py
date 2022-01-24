import os, sys
try:
    import hamiltorch
    HAMILTORCH_AVAILABLE = True
except ImportError:
    HAMILTORCH_AVAILABLE = False
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.laplace import Laplace

from advbench import perturbations

class Attack(nn.Module):
    def __init__(self, classifier, hparams, device, perturbation='Linf'):
        super(Attack, self).__init__()
        self.classifier = classifier
        self.hparams = hparams
        self.device = device
        eps = self.hparams['epsilon']
        self.perturbation = vars(perturbations)[perturbation](eps)
    def forward(self, imgs, labels):
        raise NotImplementedError

class Attack_Linf(Attack):
    def __init__(self, classifier, hparams, device, perturbation='Linf'):
        super(Attack_Linf, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
    
class PGD_Linf(Attack_Linf):
    def __init__(self, classifier, hparams, device, perturbation='Linf'):
        super(PGD_Linf, self).__init__(classifier, hparams, device, perturbation=perturbation)
    
    def forward(self, imgs, labels):
        self.classifier.eval()
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        for _ in range(self.hparams['pgd_n_steps']):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            delta += self.hparams['pgd_step_size']* torch.sign(grad)
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)
            
        self.classifier.train()
        return adv_imgs.detach(), delta.detach()    # this detach may not be necessary

class TRADES_Linf(Attack_Linf):
    def __init__(self, classifier, hparams,  device, perturbation='Linf'):
        super(TRADES_Linf, self).__init__(classifier, hparams, device, perturbation=perturbation)
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')  # AR: let's write a method to do the log-softmax part

    def forward(self, imgs, labels):
        self.classifier.eval()
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        for _ in range(self.hparams['trades_n_steps']):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = self.kl_loss_fn(
                    F.log_softmax(self.classifier(adv_imgs), dim=1),   # AR: Note that this means that we can't have softmax at output of classifier
                    F.softmax(self.classifier(imgs), dim=1))
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            delta += self.hparams['trades_step_size']* torch.sign(grad)
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)
        
        self.classifier.train()
        return adv_imgs.detach(), delta.detach() # this detach may not be necessary

class FGSM_Linf(Attack):
    def __init__(self, classifier,  hparams, device,  perturbation='Linf'):
        super(FGSM_Linf, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def forward(self, imgs, labels):
        self.classifier.eval()

        imgs.requires_grad = True
        adv_loss = F.cross_entropy(self.classifier(imgs), labels)
        grad = torch.autograd.grad(adv_loss, [imgs])[0].detach()
        delta = self.hparams['epsilon'] * grad.sign()
        delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()

        return adv_imgs.detach(), delta.detach()

class LMC_Gaussian_Linf(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(LMC_Gaussian_Linf, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        for _ in range(self.hparams['g_dale_n_steps']):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = torch.log(1 - torch.softmax(self.classifier(adv_imgs), dim=1)[range(batch_size), labels]).mean()
                # adv_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            noise = torch.randn_like(delta).to(self.device).detach()
            delta += self.hparams['g_dale_step_size'] * torch.sign(grad) + self.hparams['g_dale_noise_coeff'] * noise
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()

        return adv_imgs.detach(), delta.detach()

class LMC_Laplacian_Linf(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(LMC_Laplacian_Linf, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        noise_dist = Laplace(torch.tensor(0.), torch.tensor(1.))
        delta = self.perturbation.delta_init(imgs).to(imgs.device)
        for _ in range(self.hparams['l_dale_n_steps']):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = torch.log(1 - torch.softmax(self.classifier(adv_imgs), dim=1)[range(batch_size), labels]).mean()
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            noise = noise_dist.sample(grad.shape)
            delta += self.hparams['l_dale_step_size'] * torch.sign(grad + self.hparams['l_dale_noise_coeff'] * noise)
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()
        return adv_imgs.detach(), delta.detach()

class Grid_Search(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Grid_Search, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
        
        self.dim = self.perturbation.dim
        if self.dim==1:
            self.grid_shape = [self.hparams['grid_size']]
            self.epsilon = [self.hparams['epsilon']]
        else:
            grid = []
            epsilon = []
            for i in range(self.dim):
                grid.append(self.hparams[f'grid_size_{i}'])
                epsilon.append(self.hparams[f'epsilon_{i}'])
            self.grid_shape = grid
            self.epsilon = epsilon
        self.make_grid()
    
    def make_grid(self):
        if self.dim>1:
            grid = torch.empty(self.grid_shape).to(self.device)
            for idx, (eps, num) in enumerate(zip(self.grid_shape, self.epsilon)):
                step = 2*eps/num
                grid[idx] = torch.arange(-eps, eps, step=step, device=self.device)
        else:
            eps = self.epsilon[0]
            step = 2*eps/self.grid_shape[0]
            grid = torch.arange(-eps, eps, step=step, device=self.device)

        self.grid = grid
        self.grid_size = torch.numel(grid)

    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        deltas_size = [batch_size]
        for dim in self.grid_shape:
            deltas_size.append(dim)
        with torch.no_grad():
            adv_imgs, y = self.perturbation.perturb_img(
                imgs,
                self.grid,
                repeat=True,
                labels=y)
            y_hat_adv = algorithm.predict(adv_imgs)
            adv_loss = cross_entropy(y_hat_adv, y, reduction="none").reshape(*deltas_size, -1)
        max_idx = torch.argmax(adv_loss,dim=-1)
        for dim in self.grid_size:
            max_idx.append(range(dim))
        delta = self.grid[max_idx]
        adv_imgs = self.perturbation.perturb_img(imgs, delta)
        self.classifier.train()
        return adv_imgs.detach(), delta.detach()

class Worst_Of_K(Attack_Linf):
    def __init__(self, classifier,  hparams, device, perturbation='Linf'):
        super(Worst_Of_K, self).__init__(classifier,  hparams, device,  perturbation=perturbation)

    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        delta = self.perturbation.delta_init(imgs)
        deltas_size = [self.hparams['worst_of_k_steps']]
        for dim in delta.shape:
            deltas_size.append(dim)
        deltas = torch.empty(deltas_size).to(self.device)
        adv_loss = torch.empty((self.hparams['worst_of_k_steps'], imgs.shape[0]))
        for i in range(self.hparams['worst_of_k_steps']):
            with torch.no_grad():
                delta = self.perturbation.delta_init(imgs).to(imgs.device)
                delta = self.perturbation.clamp_delta(delta, imgs)
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss[i] = torch.log(1 - torch.softmax(self.classifier(adv_imgs), dim=1)[range(batch_size), labels])
                deltas[i] = delta
        max_idx = [torch.argmax(adv_loss, dim=0), range(delta.shape[0])]
        delta = deltas[max_idx]
        adv_imgs = self.perturbation.perturb_img(imgs, delta)
        self.classifier.train()
        return adv_imgs.detach(), delta.detach()

if HAMILTORCH_AVAILABLE:
    class NUTS(Attack_Linf):
        def __init__(self, classifier,  hparams, device, perturbation='Linf'):
            super(NUTS, self).__init__(classifier,  hparams, device,  perturbation=perturbation)
            self.infty = 10e8 #torch.tensor(float('inf')).to(device)
            self.burn = self.hparams['n_burn']
            self.eps = hparams['epsilon']

        def forward(self, imgs, labels):
            self.classifier.eval()
            batch_size = imgs.size(0)
            total_size = 1
            img_dims = tuple(d for d in range(1,imgs.dim()))
            for i in imgs.size():
                total_size = total_size*i
            params_init = 0.001*torch.rand(total_size).to(self.device)
            def log_prob(delta):
                delta = delta.reshape(imgs.shape)
                adv_imgs = imgs+torch.clamp(delta, min=-self.eps, max=self.eps)
                loss = 1 - torch.softmax(self.classifier(adv_imgs), dim=1)[range(batch_size), labels]
                log_loss = torch.log(loss)
                #log_loss[torch.amax(torch.abs(delta),img_dims)>self.eps] = - self.infty
                return log_loss.sum()
            self.blockPrint()
            delta = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init,
                                    num_samples=self.burn+self.hparams['n_dale_n_steps'],
                                    step_size=self.hparams['n_dale_step_size'],
                                    burn = self.burn,
                                    num_steps_per_sample=7,
                                    desired_accept_rate=0.8)[-1]
            self.enablePrint()
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            adv_imgs = imgs + delta.reshape(imgs.shape)
            self.classifier.train()
            return adv_imgs.detach(), delta.detach()
        # Disable
        def blockPrint(self):
            sys.stdout = open(os.devnull, 'w')

        # Restore
        def enablePrint(self):
            sys.stdout = sys.__stdout__

    