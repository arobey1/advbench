from kornia.geometry import rotate, translate
import torch
from advbench.lib.transformations import se_transform

class Perturbation():
    def __init__(self, epsilon):
        self.eps = epsilon
    def clamp_delta(self, delta):
        raise NotImplementedError
    def perturb_img(self, imgs, delta, repeat=False, labels = None):
        if not repeat:
            return self._perturb(imgs, delta)
        else: # apply delta to every img in batch
            x = imgs.repeat_interleave(delta.shape[0], dim=0)
            adv_imgs = self._perturb(x, delta.repeat(imgs[0]))
            if labels is not None:
                y = labels.repeat_interleave(delta.shape[0], dim=0)
                return adv_imgs, y
            else:
                return adv_imgs
    def _perturb(self, imgs, delta):
        raise NotImplementedError
    def delta_init(self, imgs):
        raise NotImplementedError

class Linf(Perturbation):
    def __init__(self, epsilon):
        super(Linf, self).__init__(epsilon)
        self.dim = None
    def clamp_delta(self, delta, imgs):
        """Clamp delta so that (1) the perturbation is bounded
        in the l_inf norm by self.hparams['epsilon'] and (2) so that the
        perturbed image is in [0, 1]^d."""
        eps = self.eps
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(delta, -imgs, 1-imgs)
        return delta

    def _perturb(self, imgs, delta):
        if self.dim is None:
            self.dim = imgs.shape[1:]
        return imgs + delta

    def delta_init(self, imgs):
        if self.dim is None:
            self.dim = imgs.shape[1:]
        return 0.001 * torch.randn(imgs.shape)
        
class Rotation(Perturbation):
    def __init__(self, epsilon):
        super(Rotation, self).__init__(epsilon)
        self.dim = 1
    def clamp_delta(self, delta, imgs):
        """Clamp delta so that (1) the perturbation is bounded
        in the l_inf norm by self.hparams['epsilon'] and (2) so that the
        perturbed image is in [0, 1]^d."""
        delta = torch.clamp(delta, -self.eps, self.eps)
        return delta

    def _perturb(self, imgs, delta):
        return rotate(imgs, delta)
        


    def delta_init(self, imgs):
        eps = self.eps
        delta_init =   2*eps* torch.rand(imgs.shape[0])-eps
        return delta_init

class SE(Perturbation):
    def __init__(self, epsilon):
        super(SE, self).__init__(epsilon)
        self.dim = 3
    def clamp_delta(self, delta, imgs):
        """Clamp delta so that (1) the perturbation is bounded
        in the l_inf norm by self.hparams['epsilon'] and (2) so that the
        perturbed image is in [0, 1]^d."""
        for i in range(self.dim):
            delta[:, i] = torch.clamp(delta[:, i], -self.eps[i], self.eps[i])
        return delta

    def _perturb(self, imgs, delta):
        return se_transform(imgs, delta)

    def delta_init(self, imgs):
        delta_init = torch.empty(imgs.shape[0], self.dim, device=imgs.device)
        for i in range(self.dim):
            eps = self.eps[i]
            delta_init[:,i] =   2*eps* torch.randn(imgs.shape[0], device = imgs.device)-eps
        return delta_init