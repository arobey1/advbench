from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10 as CIFAR10_
from torchvision.datasets import MNIST as MNIST_
from torchvision.datasets import SVHN as SVHN_
from RandAugment import RandAugment

SPLITS = ['train', 'val', 'test']
DATASETS = ['CIFAR10', 'MNIST', 'SVHN']

def to_loaders(all_datasets, hparams):
    
    def _to_loader(split, dataset):
        batch_size = hparams['batch_size'] if split == 'train' else 100
        return DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            num_workers=all_datasets.N_WORKERS,
            shuffle=False)
            # shuffle=(split == 'train'))
    
    return [_to_loader(s, d) for (s, d) in all_datasets.splits.items()]


class AdvRobDataset(Dataset):

    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    NUM_CLASSES = None       # Subclasses should override
    N_EPOCHS = None          # Subclasses should override
    CHECKPOINT_FREQ = None   # Subclasses should override
    LOG_INTERVAL = None      # Subclasses should override
    HAS_LR_SCHEDULE = False  # Default, subclass may override

    def __init__(self):
        self.splits = dict.fromkeys(SPLITS)


class CIFAR10(AdvRobDataset):
 
    INPUT_SHAPE = (3, 32, 32)
    NUM_CLASSES = 10
    N_EPOCHS = 115
    CHECKPOINT_FREQ = 10
    LOG_INTERVAL = 100
    HAS_LR_SCHEDULE = True

    # test adversary parameters
    ADV_STEP_SIZE = 2/255.
    N_ADV_STEPS = 20

    def __init__(self, root):
        super(CIFAR10, self).__init__()

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

class MNIST(AdvRobDataset):

    INPUT_SHAPE = (1, 28, 28)
    NUM_CLASSES = 10
    N_EPOCHS = 150
    CHECKPOINT_FREQ = 10
    LOG_INTERVAL = 100
    HAS_LR_SCHEDULE = False

    # test adversary parameters
    ADV_STEP_SIZE = 0.1
    N_ADV_STEPS = 10

    def __init__(self, root):
        super(MNIST, self).__init__()
        
        xforms = transforms.ToTensor()

        train_data = MNIST_(root, train=True, transform=xforms)
        self.splits['train'] = train_data
        # self.splits['train'] = Subset(train_data, range(60000))

        train_data = MNIST_(root, train=True, transform=xforms)
        self.splits['val'] = Subset(train_data, range(54000, 60000))

        self.splits['test'] = MNIST_(root, train=False, transform=xforms)

    @staticmethod
    def adjust_lr(optimizer, epoch, hparams):

        lr = hparams['learning_rate']
        if epoch >= 55:
            lr = hparams['learning_rate'] * 0.1
        if epoch >= 75:
            lr = hparams['learning_rate'] * 0.01
        if epoch >= 90:
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

    def __init__(self, root):
        super(SVHN, self).__init__()

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