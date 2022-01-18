from kornia.geometry import rotate
import torch


class Perturbation:
    def __init__(self, epsilon):
        self.eps = epsilon

    def clamp_delta(self, delta):
        raise NotImplementedError

    def perturb_img(self, imgs, delta):
        raise NotImplementedError

    def delta_init(self, imgs):
        raise NotImplementedError


class Linf(Perturbation):
    def __init__(self, epsilon):
        super(Linf, self).__init__(epsilon)

    def clamp_delta(self, delta, imgs):
        """Clamp delta so that (1) the perturbation is bounded
        in the l_inf norm by self.hparams['epsilon'] and (2) so that the
        perturbed image is in [0, 1]^d."""
        eps = self.eps
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(delta, -imgs, 1 - imgs)
        return delta

    def perturb_img(self, imgs, delta):
        return imgs + delta

    def delta_init(self, imgs):
        return 0.001 * torch.randn(imgs.shape)


class Rotation(Perturbation):
    def __init__(self, epsilon):
        super(Rotation, self).__init__(epsilon)

    def clamp_delta(self, delta, imgs):
        """Clamp delta so that (1) the perturbation is bounded
        in the l_inf norm by self.hparams['epsilon'] and (2) so that the
        perturbed image is in [0, 1]^d."""
        delta = torch.clamp(delta, -self.eps, self.eps)
        return delta

    def perturb_img(self, imgs, delta):
        return rotate(imgs, delta)

    def delta_init(self, imgs):
        eps = self.eps
        delta_init = 2 * eps * torch.randn(imgs.shape[0]) - eps
        return delta_init
