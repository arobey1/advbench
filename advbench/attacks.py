import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.laplace import Laplace

class Attack(nn.Module):
    def __init__(self, classifier, hparams, device):
        super(Attack, self).__init__()
        self.classifier = classifier
        self.hparams = hparams
        self.device = device

    def forward(self, imgs, labels):
        raise NotImplementedError

class Attack_Linf(Attack):
    def __init__(self, classifier, hparams, device):
        super(Attack_Linf, self).__init__(classifier, hparams, device)
    
    def _clamp_perturbation(self, imgs, adv_imgs):
        """Clamp a perturbed image so that (1) the perturbation is bounded
        in the l_inf norm by self.hparams['epsilon'] and (2) so that the
        perturbed image is in [0, 1]^d."""

        eps = self.hparams['epsilon']
        adv_imgs = torch.min(torch.max(adv_imgs, imgs - eps), imgs + eps)
        return torch.clamp(adv_imgs, 0.0, 1.0)

class PGD_Linf(Attack_Linf):
    def __init__(self, classifier, hparams, device):
        super(PGD_Linf, self).__init__(classifier, hparams, device)
    
    def forward(self, imgs, labels):
        self.classifier.eval()

        adv_imgs = imgs.detach() # + 0.001 * torch.randn(imgs.shape).to(self.device).detach() #AR: is this detach necessary?
        for _ in range(self.hparams['pgd_n_steps']):
            adv_imgs.requires_grad_(True)
            with torch.enable_grad():
                adv_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
            grad = torch.autograd.grad(adv_loss, [adv_imgs])[0].detach()
            adv_imgs = adv_imgs + self.hparams['pgd_step_size']* torch.sign(grad)
            adv_imgs = self._clamp_perturbation(imgs, adv_imgs)
            
        self.classifier.train()
        return adv_imgs.detach()    # this detach may not be necessary

class SmoothAdv(Attack_Linf):
    def __init__(self, classifier, hparams, device):
        super(SmoothAdv, self).__init__(classifier, hparams, device)

    def sample_deltas(self, imgs):
        sigma = self.hparams['rand_smoothing_sigma']
        return sigma * torch.randn_like(imgs)
    
    def forward(self, imgs, labels):
        self.classifier.eval()

        adv_imgs = imgs.detach()
        for _ in range(self.hparams['rand_smoothing_n_steps']):
            adv_imgs.requires_grad_(True)
            loss = 0.
            for _ in range(self.hparams['rand_smoothing_n_samples']):
                deltas = self.sample_deltas(imgs)
                loss += F.softmax(self.classifier(adv_imgs + deltas), dim=1)[range(imgs.size(0)), labels]

            total_loss = -1. * torch.log(loss / self.hparams['rand_smoothing_n_samples']).mean()
            grad = torch.autograd.grad(total_loss, [adv_imgs])[0].detach()
            adv_imgs = imgs + self.hparams['rand_smoothing_step_size'] * torch.sign(grad)
            adv_imgs = self._clamp_perturbation(imgs, adv_imgs)

        self.classifier.train()
        return adv_imgs.detach()    # this detach may not be necessary


class TRADES_Linf(Attack_Linf):
    def __init__(self, classifier, hparams, device):
        super(TRADES_Linf, self).__init__(classifier, hparams, device)
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')  # AR: let's write a method to do the log-softmax part

    def forward(self, imgs, labels):
        self.classifier.eval()

        adv_imgs = imgs.detach() + 0.001 * torch.randn(imgs.shape).to(self.device).detach()  #AR: is this detach necessary?
        for _ in range(self.hparams['trades_n_steps']):
            adv_imgs.requires_grad_(True)
            with torch.enable_grad():
                adv_loss = self.kl_loss_fn(
                    F.log_softmax(self.classifier(adv_imgs), dim=1),   # AR: Note that this means that we can't have softmax at output of classifier
                    F.softmax(self.classifier(imgs), dim=1))
            
            grad = torch.autograd.grad(adv_loss, [adv_imgs])[0].detach()
            adv_imgs = adv_imgs + self.hparams['trades_step_size']* torch.sign(grad)
            adv_imgs = self._clamp_perturbation(imgs, adv_imgs)
        
        self.classifier.train()
        return adv_imgs.detach() # this detach may not be necessary

class FGSM_Linf(Attack):
    def __init__(self, classifier, hparams, device):
        super(FGSM_Linf, self).__init__(classifier, hparams, device)

    def forward(self, imgs, labels):
        self.classifier.eval()

        imgs.requires_grad = True
        adv_loss = F.cross_entropy(self.classifier(imgs), labels)
        grad = torch.autograd.grad(adv_loss, [imgs])[0].detach()
        adv_imgs = imgs + self.hparams['epsilon'] * grad.sign()
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        self.classifier.train()

        return adv_imgs.detach()

class LMC_Gaussian_Linf(Attack_Linf):
    def __init__(self, classifier, hparams, device):
        super(LMC_Gaussian_Linf, self).__init__(classifier, hparams, device)

    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)

        adv_imgs = imgs.detach() + 0.001 * torch.randn(imgs.shape).to(self.device).detach() #AR: is this detach necessary?
        for _ in range(self.hparams['g_dale_n_steps']):
            adv_imgs.requires_grad_(True)
            with torch.enable_grad():
                adv_loss = torch.log(1 - torch.softmax(self.classifier(adv_imgs), dim=1)[range(batch_size), labels]).mean()
                # adv_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
            grad = torch.autograd.grad(adv_loss, [adv_imgs])[0].detach()
            noise = torch.randn_like(adv_imgs).to(self.device).detach()

            adv_imgs = adv_imgs + self.hparams['g_dale_step_size'] * torch.sign(grad) + self.hparams['g_dale_noise_coeff'] * noise
            adv_imgs = self._clamp_perturbation(imgs, adv_imgs)
            
        self.classifier.train()

        return adv_imgs.detach()

class LMC_Laplacian_Linf(Attack_Linf):
    def __init__(self, classifier, hparams, device):
        super(LMC_Laplacian_Linf, self).__init__(classifier, hparams, device)

    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        noise_dist = Laplace(torch.tensor(0.), torch.tensor(1.))

        adv_imgs = imgs.detach() + 0.001 * torch.randn(imgs.shape).to(self.device).detach() #AR: is this detach necessary?
        for _ in range(self.hparams['l_dale_n_steps']):
            adv_imgs.requires_grad_(True)
            with torch.enable_grad():
                adv_loss = torch.log(1 - torch.softmax(self.classifier(adv_imgs), dim=1)[range(batch_size), labels]).mean()
            grad = torch.autograd.grad(adv_loss, [adv_imgs])[0].detach()
            noise = noise_dist.sample(grad.shape)
            adv_imgs = adv_imgs + self.hparams['l_dale_step_size'] * torch.sign(grad + self.hparams['l_dale_noise_coeff'] * noise)
            adv_imgs = self._clamp_perturbation(imgs, adv_imgs)

        self.classifier.train()
        return adv_imgs.detach()