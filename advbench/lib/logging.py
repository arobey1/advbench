import torch
from torch.nn.functional import cross_entropy
from advbench import perturbations

class PerturbationEval():
    def __init__(self, perturbation, num_batches, num_perturbations, epsilon, device, dim=1):
        self.perturbation = vars(perturbations)[perturbation](epsilon)
        self.num_batches = num_batches
        self.num_perturbations = num_perturbations
        self.epsilon = epsilon
        self.device = device
        self.dim = dim
        self.fixed_deltas = False
    
    def eval_perturbed(self, algorithm, loader):
            losses = torch.empty(self.num_batches, self.num_perturbations).to(self.device)
            if not self.fixed_deltas:
                all_deltas = torch.empty(self.num_batches, self.num_perturbations).to(self.device)
            algorithm.eval()
            with torch.no_grad():
                for idx, (x, y) in enumerate(loader):
                    if idx >= self.num_batches:
                        # limit batches
                        break
                    else:
                        deltas = self.get_delta(x)
                        if not self.fixed_deltas:
                            all_deltas[idx] = deltas
                        # Compute Loss on transformed data (constraint)
                        batch_size = x.shape[0]
                        x_rot = x.to(self.device).repeat_interleave(
                            self.num_perturbations, dim=0
                        )
                        x_aug = self.perturbation.perturb_img(
                            x_rot,
                            deltas)
                        y = y.to(self.device).repeat_interleave(
                            self.num_perturbations, dim=0
                        )
                        y_hat_aug = algorithm.predict(x_aug)
                        aug_loss = cross_entropy(y_hat_aug, y, reduction="none")
                        losses[idx] = aug_loss.reshape((batch_size, -1)).mean(dim=0)
                    if self.fixed_deltas:
                        all_deltas = self.grid
            algorithm.train()
            return losses.mean(dim=0), all_deltas

    def get_delta(self, idx, imgs):
        pass

class GridEval(PerturbationEval):
    def __init__(self,perturbation, num_batches, num_perturbations, epsilon, device):
        super(GridEval, self).__init__(perturbation, num_batches, num_perturbations, epsilon, device)
        self.make_grid()
        self.fixed_deltas = True
        self.delta_shape = self.grid.shape
    def make_grid(self):
        if self.dim == 1:
            eps = self.epsilon
            step = 2*eps/self.num_perturbations
            self.grid = torch.arange(-eps, eps, step=step, device = self.device)
        else:
            # Do some Dim reduction (ie PCA)
            raise NotImplementedError
    
    def get_delta(self, imgs):
        return self.grid.repeat(imgs.shape[0])