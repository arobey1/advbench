import os, sys
import hamiltorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.laplace import Laplace

from advbench import perturbations


class Attack(nn.Module):
    def __init__(self, classifier, hparams, device, perturbation="Linf"):
        super(Attack, self).__init__()
        self.classifier = classifier
        self.hparams = hparams
        self.device = device
        eps = self.hparams["epsilon"]
        self.perturbation = vars(perturbations)[perturbation](eps)

    def forward(self, imgs, labels):
        raise NotImplementedError


class Attack_Linf(Attack):
    def __init__(self, classifier, hparams, device, perturbation="Linf"):
        super(Attack_Linf, self).__init__(
            classifier, hparams, device, perturbation=perturbation
        )


class PGD_Linf(Attack_Linf):
    def __init__(self, classifier, hparams, device, perturbation="Linf"):
        super(PGD_Linf, self).__init__(
            classifier, hparams, device, perturbation=perturbation
        )

    def forward(self, imgs, labels):
        self.classifier.eval()
        delta = self.perturbation.delta_init(imgs).to(self.device)
        for _ in range(self.hparams["pgd_n_steps"]):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            delta += self.hparams["pgd_step_size"] * torch.sign(grad)
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()
        return adv_imgs.detach(), delta.detach()  # this detach may not be necessary


class TRADES_Linf(Attack_Linf):
    def __init__(self, classifier, hparams, device, perturbation="Linf"):
        super(TRADES_Linf, self).__init__(
            classifier, hparams, device, perturbation=perturbation
        )
        self.kl_loss_fn = nn.KLDivLoss(
            reduction="batchmean"
        )  # AR: let's write a method to do the log-softmax part

    def forward(self, imgs, labels):
        self.classifier.eval()
        delta = self.perturbation.delta_init(imgs).to(self.device)
        for _ in range(self.hparams["trades_n_steps"]):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = self.kl_loss_fn(
                    F.log_softmax(
                        self.classifier(adv_imgs), dim=1
                    ),  # AR: Note that this means that we can't have softmax at output of classifier
                    F.softmax(self.classifier(imgs), dim=1),
                )
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            delta += self.hparams["trades_step_size"] * torch.sign(grad)
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()
        return adv_imgs.detach(), delta.detach()  # this detach may not be necessary


class FGSM_Linf(Attack):
    def __init__(self, classifier, hparams, device, perturbation="Linf"):
        super(FGSM_Linf, self).__init__(
            classifier, hparams, device, perturbation=perturbation
        )

    def forward(self, imgs, labels):
        self.classifier.eval()

        imgs.requires_grad = True
        adv_loss = F.cross_entropy(self.classifier(imgs), labels)
        grad = torch.autograd.grad(adv_loss, [imgs])[0].detach()
        delta = self.hparams["epsilon"] * grad.sign()
        delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()

        return adv_imgs.detach(), delta.detach()


class LMC_Gaussian_Linf(Attack_Linf):
    def __init__(self, classifier, hparams, device, perturbation="Linf"):
        super(LMC_Gaussian_Linf, self).__init__(
            classifier, hparams, device, perturbation=perturbation
        )

    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        delta = self.perturbation.delta_init(imgs).to(self.device)
        for _ in range(self.hparams["g_dale_n_steps"]):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = torch.log(
                    1
                    - torch.softmax(self.classifier(adv_imgs), dim=1)[
                        range(batch_size), labels
                    ]
                ).mean()
                # adv_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            noise = torch.randn_like(delta).to(self.device).detach()
            delta += (
                self.hparams["g_dale_step_size"] * torch.sign(grad)
                + self.hparams["g_dale_noise_coeff"] * noise
            )
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()

        return adv_imgs.detach(), delta.detach()


class LMC_Laplacian_Linf(Attack_Linf):
    def __init__(self, classifier, hparams, device, perturbation="Linf"):
        super(LMC_Laplacian_Linf, self).__init__(
            classifier, hparams, device, perturbation=perturbation
        )

    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        noise_dist = Laplace(torch.tensor(0.0), torch.tensor(1.0))
        delta = self.perturbation.delta_init(imgs).to(self.device)
        for _ in range(self.hparams["l_dale_n_steps"]):
            delta.requires_grad_(True)
            with torch.enable_grad():
                adv_imgs = self.perturbation.perturb_img(imgs, delta)
                adv_loss = torch.log(
                    1
                    - torch.softmax(self.classifier(adv_imgs), dim=1)[
                        range(batch_size), labels
                    ]
                ).mean()
            grad = torch.autograd.grad(adv_loss, [delta])[0].detach()
            delta.requires_grad_(False)
            noise = noise_dist.sample(grad.shape)
            delta += self.hparams["l_dale_step_size"] * torch.sign(
                grad + self.hparams["l_dale_noise_coeff"] * noise
            )
            delta = self.perturbation.clamp_delta(delta, imgs)
        adv_imgs = self.perturbation.perturb_img(imgs, delta)

        self.classifier.train()
        return adv_imgs.detach(), delta.detach()


class NUTS(Attack_Linf):
    def __init__(self, classifier, hparams, device, perturbation="Linf"):
        super(NUTS, self).__init__(
            classifier, hparams, device, perturbation=perturbation
        )
        self.infty = 10e8  # torch.tensor(float('inf')).to(device)
        self.burn = self.hparams["n_burn"]
        self.eps = hparams["epsilon"]

    def forward(self, imgs, labels):
        self.classifier.eval()
        batch_size = imgs.size(0)
        total_size = 1
        img_dims = tuple(d for d in range(1, imgs.dim()))
        for i in imgs.size():
            total_size = total_size * i
        params_init = 0.001 * torch.rand(total_size).to(self.device)

        def log_prob(delta):
            delta = delta.reshape(imgs.shape)
            adv_imgs = imgs + torch.clamp(delta, min=-self.eps, max=self.eps)
            loss = (
                1
                - torch.softmax(self.classifier(adv_imgs), dim=1)[
                    range(batch_size), labels
                ]
            )
            log_loss = torch.log(loss)
            # log_loss[torch.amax(torch.abs(delta),img_dims)>self.eps] = - self.infty
            return log_loss.sum()

        self.blockPrint()
        delta = hamiltorch.sample(
            log_prob_func=log_prob,
            params_init=params_init,
            num_samples=self.burn + self.hparams["n_dale_n_steps"],
            step_size=self.hparams["n_dale_step_size"],
            burn=self.burn,
            num_steps_per_sample=7,
            desired_accept_rate=0.8,
        )[-1]
        self.enablePrint()
        delta = torch.clamp(delta, min=-self.eps, max=self.eps)
        adv_imgs = imgs + delta.reshape(imgs.shape)
        self.classifier.train()
        return adv_imgs.detach(), delta.detach()

    # Disable
    def blockPrint(self):
        sys.stdout = open(os.devnull, "w")

    # Restore
    def enablePrint(self):
        sys.stdout = sys.__stdout__
