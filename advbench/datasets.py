import torch
from torch.utils.data import Subset, ConcatDataset, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10 as CIFAR10_
from torchvision.datasets import MNIST as TorchvisionMNIST
from torchvision.datasets import SVHN as SVHN_
from RandAugment import RandAugment

SPLITS = ['train', 'val', 'test']
DATASETS = ['CIFAR10', 'MNIST', 'SVHN']

class AdvRobDataset:

    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    NUM_CLASSES = None       # Subclasses should override
    N_EPOCHS = None          # Subclasses should override
    CHECKPOINT_FREQ = None   # Subclasses should override
    LOG_INTERVAL = None      # Subclasses should override
    HAS_LR_SCHEDULE = False  # Default, subclass may override
    ON_DEVICE = False        # Default, subclass may override

    def __init__(self, device):
        self.splits = dict.fromkeys(SPLITS)
        self.device = device

class CIFAR10(AdvRobDataset):
 
    INPUT_SHAPE = (3, 32, 32)
    NUM_CLASSES = 10
    N_EPOCHS = 115
    CHECKPOINT_FREQ = 10
    LOG_INTERVAL = 100
    HAS_LR_SCHEDULE = True

    def __init__(self, root, device):
        super(CIFAR10, self).__init__(device)

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        test_transforms = transforms.ToTensor()

        train_data = CIFAR10_(root, train=True, transform=train_transforms)
        self.splits['train'] = train_data
        # self.splits['train'] = Subset(train_data, range(5000))

        train_data = CIFAR10_(root, train=True, transform=train_transforms)
        self.splits['val'] = Subset(train_data, range(45000, 50000))

        self.splits['test'] = CIFAR10_(root, train=False, transform=test_transforms)

    @staticmethod
    def adjust_lr(optimizer, epoch, hparams):
        lr = hparams['learning_rate']
        if epoch >= 55:    # 150
            lr = hparams['learning_rate'] * 0.1
        if epoch >= 75:    # 175
            lr = hparams['learning_rate'] * 0.01
        if epoch >= 90:    # 190
            lr = hparams['learning_rate'] * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class MNISTTensor(AdvRobDataset):

    N_WORKERS = 0       # Needs to be zero so we don't fetch from GPU
    INPUT_SHAPE = (1, 28, 28)
    NUM_CLASSES = 10
    N_EPOCHS = 50
    CHECKPOINT_FREQ = 10
    LOG_INTERVAL = 100
    HAS_LR_SCHEDULE = True
    ON_DEVICE = True

    def __init__(self, root, device):
        super(MNISTTensor, self).__init__(device)
        
        train_data = TorchvisionMNIST(
            root=root, 
            train=True, 
            transform=transforms.ToTensor())
        test_data = TorchvisionMNIST(
            root=root,
            train=False,
            transform=transforms.ToTensor())

        all_imgs = torch.cat((
            train_data.data, 
            test_data.data)).reshape(-1, 1, 28, 28).float().to(self.device)
        all_labels = torch.cat((
            train_data.targets, 
            test_data.targets)).to(self.device)

        self.splits = {
            'train': TensorDataset(all_imgs, all_labels),
            'validation': TensorDataset(all_imgs, all_labels),
            'test': TensorDataset(all_imgs, all_labels)
        }

    @staticmethod
    def adjust_lr(optimizer, epoch, hparams):

        lr = hparams['learning_rate']
        if epoch >= 25:
            lr = hparams['learning_rate'] * 0.1
        if epoch >= 35:
            lr = hparams['learning_rate'] * 0.01
        if epoch >= 40:
            lr = hparams['learning_rate'] * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class MNIST(AdvRobDataset):

    INPUT_SHAPE = (1, 28, 28)
    NUM_CLASSES = 10
    N_EPOCHS = 50
    CHECKPOINT_FREQ = 10
    LOG_INTERVAL = 100
    HAS_LR_SCHEDULE = True

    def __init__(self, root, device):
        super(MNIST, self).__init__(device)
        
        train_data = TorchvisionMNIST(
            root=root, 
            train=True, 
            transform=transforms.ToTensor())
        test_data = TorchvisionMNIST(
            root=root,
            train=False,
            transform=transforms.ToTensor())

        # self.splits = {
        #     'train': Subset(train_data, range(54000)),
        #     'validation': Subset(train_data, range(54000, 60000)),
        #     'test': test_data
        # }

        all_data = ConcatDataset([train_data, test_data])
        self.splits = {
            'train': all_data,
            'validation': all_data,
            'test': all_data
        }

    @staticmethod
    def adjust_lr(optimizer, epoch, hparams):

        lr = hparams['learning_rate']
        if epoch >= 25:
            lr = hparams['learning_rate'] * 0.1
        if epoch >= 35:
            lr = hparams['learning_rate'] * 0.01
        if epoch >= 40:
            lr = hparams['learning_rate'] * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class SVHN(AdvRobDataset):
     
    INPUT_SHAPE = (3, 32, 32)
    NUM_CLASSES = 10
    N_EPOCHS = 115
    CHECKPOINT_FREQ = 10
    LOG_INTERVAL = 100
    HAS_LR_SCHEDULE = False

    def __init__(self, root, device):
        super(SVHN, self).__init__(device)

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        test_transforms = transforms.ToTensor()

        train_data = SVHN_(root, split='train', transform=train_transforms, download=True)
        self.splits['train'] = train_data
        self.splits['test'] = SVHN_(root, split='test', transform=test_transforms, download=True)

    @staticmethod
    def adjust_lr(optimizer, epoch, hparams):
        lr = hparams['learning_rate']
        if epoch >= 55:    # 150
            lr = hparams['learning_rate'] * 0.1
        if epoch >= 75:    # 175
            lr = hparams['learning_rate'] * 0.01
        if epoch >= 90:    # 190
            lr = hparams['learning_rate'] * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr