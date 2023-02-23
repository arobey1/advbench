import torch
import torch.nn.functional as F

from advbench import attacks

class Evaluator:

    # Sub-class should over-ride
    NAME = ''   

    def __init__(self, algorithm, device, test_hparams):
        self.algorithm = algorithm
        self.device = device
        self.test_hparams = test_hparams

    def calculate(self, loader):
        raise NotImplementedError

    def sample_perturbations(self, imgs):
        eps = self.test_hparams['epsilon']
        return 2 * eps * torch.rand_like(imgs) - eps

    @staticmethod
    def clamp_imgs(imgs):
        return torch.clamp(imgs, 0.0, 1.0)

class Clean(Evaluator):
    """Calculates the standard accuracy of a classifier."""

    NAME = 'Clean'

    def __init__(self, algorithm, device, test_hparams):
        super(Clean, self).__init__(algorithm, device, test_hparams)

    @torch.no_grad()
    def calculate(self, loader):
        self.algorithm.eval()

        correct, total, loss_sum = 0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.algorithm.predict(imgs)
            loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
            preds = logits.argmax(dim=1, keepdim=True)
            correct += preds.eq(labels.view_as(preds)).sum().item()
            total += imgs.size(0)

        self.algorithm.train()
        return {
            f'{self.NAME}-Accuracy': 100. * correct / total,
            f'{self.NAME}-Loss': loss_sum / total
        }

class Adversarial(Evaluator):
    """Calculates the adversarial accuracy of a classifier."""

    def __init__(self, algorithm, device, attack, test_hparams):
        super(Adversarial, self).__init__(algorithm, device, test_hparams)
        self.attack = attack

    def calculate(self, loader):
        self.algorithm.eval()

        correct, total, loss_sum = 0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            adv_imgs = self.attack(imgs, labels)

            with torch.no_grad():
                logits = self.algorithm.predict(adv_imgs)
                loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()

            preds = logits.argmax(dim=1, keepdim=True)
            correct += preds.eq(labels.view_as(preds)).sum().item()
            total += imgs.size(0)

        self.algorithm.train()
        return {
            f'{self.NAME}-Accuracy': 100. * correct / total,
            f'{self.NAME}-Loss': float(loss_sum) / total
        }

class PGD(Adversarial):
    """Calculates the PGD adversarial accuracy of a classifier."""

    NAME = 'PGD'

    def __init__(self, algorithm, device, test_hparams):

        attack = attacks.PGD_Linf(
            classifier=algorithm.classifier,
            hparams=test_hparams,
            device=device)
        super(PGD, self).__init__(
            algorithm=algorithm, 
            device=device, 
            attack=attack, 
            test_hparams=test_hparams)

class FGSM(Adversarial):
    """Calculates the FGSM adversarial accuracy of a classifier."""

    NAME = 'FGSM'

    def __init__(self, algorithm, device, test_hparams):

        attack = attacks.FGSM_Linf(
            classifier=algorithm.classifier,
            hparams=test_hparams,
            device=device)
        super(FGSM, self).__init__(
            algorithm=algorithm, 
            device=device, 
            attack=attack, 
            test_hparams=test_hparams)

class CVaR(Evaluator):
    """Calculates the CVaR loss of a classifier."""

    NAME = 'CVaR'

    def __init__(self, algorithm, device, test_hparams):
        super(CVaR, self).__init__(algorithm, device, test_hparams)
        self.q = self.test_hparams['cvar_sgd_beta']
        self.n_cvar_steps = self.test_hparams['cvar_sgd_n_steps']
        self.M = self.test_hparams['cvar_sgd_M']
        self.step_size = self.test_hparams['cvar_sgd_t_step_size']

    @torch.no_grad()
    def calculate(self, loader):
        self.algorithm.eval()

        loss_sum, total = 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            ts = torch.zeros(size=(imgs.size(0),)).to(self.device)

            # perform n steps of optimization to compute inner inf
            for _ in range(self.n_cvar_steps):

                cvar_loss, indicator_sum = 0, 0

                # number of samples in innner expectation in def. of CVaR
                for _ in range(self.M):
                    perturbations = self.sample_perturbations(imgs)
                    perturbed_imgs = self.clamp_imgs(imgs + perturbations)
                    preds = self.algorithm.predict(perturbed_imgs)
                    loss = F.cross_entropy(preds, labels, reduction='none')

                    indicator_sum += torch.where(
                        loss > ts, 
                        torch.ones_like(ts), 
                        torch.zeros_like(ts))
                    cvar_loss += F.relu(loss - ts)

                indicator_avg = indicator_sum / float(self.M)
                cvar_loss = (ts + cvar_loss / (self.M * self.q)).mean()

                # gradient update on ts
                grad_ts = (1 - (1 / self.q) * indicator_avg) / float(imgs.size(0))
                ts = ts - self.step_size * grad_ts

            loss_sum += cvar_loss.item() * imgs.size(0)
            total += imgs.size(0)
            
        self.algorithm.train()
        
        return {f'{self.NAME}-Loss': loss_sum / float(total)}

class Augmented(Evaluator):
    """Calculates the augmented accuracy of a classifier."""

    NAME = 'Augmented'

    def __init__(self, algorithm, device, test_hparams):
        super(Augmented, self).__init__(algorithm, device, test_hparams)
        self.n_aug_samples = self.test_hparams['aug_n_samples']

    @staticmethod
    def quantile_accuracy(q, accuracy_per_datum):
        """Calculate q-Quantile accuracy"""

        # quantile predictions for each data point
        beta_quantile_acc_per_datum = torch.where(
            accuracy_per_datum > (1 - q) * 100.,
            100. * torch.ones_like(accuracy_per_datum),
            torch.zeros_like(accuracy_per_datum))

        return beta_quantile_acc_per_datum.mean().item()

    @torch.no_grad()
    def calculate(self, loader):
        self.algorithm.eval()

        correct, total, loss_sum = 0, 0, 0
        correct_per_datum = []

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            batch_correct_ls = []
            for _ in range(self.n_aug_samples):
                perturbations = self.sample_perturbations(imgs)
                perturbed_imgs = self.clamp_imgs(imgs + perturbations)
                logits = self.algorithm.predict(perturbed_imgs)
                loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
                preds = logits.argmax(dim=1, keepdim=True)

                # unreduced predictions
                pert_preds = preds.eq(labels.view_as(preds))

                # list of predictions for each data point
                batch_correct_ls.append(pert_preds)

                correct += pert_preds.sum().item()
                total += imgs.size(0)

            # number of correct predictions for each data point
            batch_correct = torch.sum(torch.hstack(batch_correct_ls), dim=1)
            correct_per_datum.append(batch_correct)

        # accuracy for each data point
        accuracy_per_datum = 100. * torch.hstack(correct_per_datum) / self.n_aug_samples

        self.algorithm.train()

        return_dict = {
            f'{self.NAME}-Accuracy': 100. * correct / total,
            f'{self.NAME}-Loss': loss_sum / total
        }

        if self.test_hparams['test_betas']:
            return_dict.update({
                f'{self.NAME}-{q}-Quantile-Accuracy': self.quantile_accuracy(q, accuracy_per_datum)
                for q in self.test_hparams['test_betas']
            })

        return return_dict

