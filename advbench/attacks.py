import torch
import torch.nn as nn
import torch.nn.functional as F

class Attack(nn.Module):
    def __init__(self, classifier, hparams):
        super(Attack, self).__init__()
        self.classifier = classifier
        self.hparams = hparams

    def forward(self, imgs, labels):
        raise NotImplementedError

class Attack_Linf(Attack):
    def __init__(self, classifier, hparams):
        super(Attack_Linf, self).__init__(classifier, hparams)
    
    def _clamp_perturbation(self, imgs, adv_imgs):
        """Clamp a perturbed image so that (1) the perturbation is bounded
        in the l_inf norm by self.hparams['epsilon'] and (2) so that the
        perturbed image is in [0, 1]^d."""

        eps = self.hparams['epsilon']
        adv_imgs = torch.min(torch.max(adv_imgs, imgs - eps), imgs + eps)
        return torch.clamp(adv_imgs, 0.0, 1.0)

class PGD_Linf(Attack_Linf):
    def __init__(self, classifier, hparams):
        super(Attack_Linf, self).__init__(classifier, hparams)
    
    def forward(self, imgs, labels):
        self.classifier.eval()

        adv_imgs = imgs.detach() + 0.001 * torch.randn(imgs.shape).cuda().detach() #AR: is this detach necessary?
        for _ in range(self.hparams['pgd_n_steps']):
            adv_imgs.requires_grad_(True)
            with torch.enable_grad():
                adv_loss = F.cross_entropy(self.classifier(adv_imgs), labels)
            grad = torch.autograd.grad(adv_loss, [adv_imgs])[0].detach()
            adv_imgs = adv_imgs + self.hparams['pgd_step_size']* torch.sign(grad)
            adv_imgs = self._clamp_perturbation(imgs, adv_imgs)
            
        self.classifier.train()
        return adv_imgs.detach()    # this detach may not be necessary

class TRADES_Linf(Attack_Linf):
    def __init__(self, classifier, hparams):
        super(Attack_Linf, self).__init__(classifier, hparams)
        self.kl_loss_fn = nn.KLDivLoss(size_average=False)  # AR: let's write a method to do the log-softmax part

    def forward(self, imgs, labels):
        self.classifier.eval()

        adv_imgs = imgs.detach() + 0.001 * torch.randn(imgs.shape).cuda().detach()  #AR: is this detach necessary?
        for _ in range(self.hparams['trades_n_steps']):
            adv_imgs.requires_grad_(True)
            with torch.enable_grad():
                adv_loss = self.kl_loss_fn(
                    F.log_softmax(self.classifier(adv_imgs), dim=1),   # AR: Note that this means that we can't have softmax at output of classifier
                    F.softmax(self.classifier(imgs), dim=1))
            
            grad = torch.autograd.grad(adv_loss, [adv_imgs])[0].detach()
            adv_imgs = adv_imgs + self.hparams['pgd_step_size']* torch.sign(grad)
            adv_imgs = self._clamp_perturbation(imgs, adv_imgs)
        
        self.classifier.train()
        return adv_imgs.detach() # this detach may not be necessary