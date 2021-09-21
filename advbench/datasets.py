from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10 as CIFAR10_
from torchvision.datasets import MNIST as MNIST_

SPLITS = ['train', 'val', 'test']

def to_loaders(all_datasets, hparams):
    
    def _to_loader(split, dataset):
        batch_size = hparams['batch_size'] if split == 'train' else 64
        return DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            num_workers=all_datasets.N_WORKERS,
            shuffle=(split == 'train'))
    
    return [_to_loader(s, d) for (s, d) in all_datasets.splits.items()]


class AdvRobDataset(Dataset):

    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    NUM_CLASSES = None       # Subclasses should override
    N_EPOCHS = None          # Subclasses should override
    CHECKPOINT_FREQ = None   # Subclasses should override

    def __init__(self):
        self.splits = dict.fromkeys(SPLITS)


class CIFAR10(AdvRobDataset):
 
    INPUT_SHAPE = (3, 32, 32)
    NUM_CLASSES = 10
    N_EPOCHS = 20

    def __init__(self, root):
        super(CIFAR10, self).__init__()

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        test_transforms = transforms.ToTensor()

        train_data = CIFAR10_(root, train=True, transform=train_transforms)
        self.splits['train'] = Subset(train_data, range(45000))

        train_data = CIFAR10_(root, train=True, transform=train_transforms)
        self.splits['val'] = Subset(train_data, range(45000, 50000))

        self.splits['test'] = CIFAR10_(root, train=False, transform=test_transforms)


class MNIST(AdvRobDataset):

    INPUT_SHAPE = (1, 28, 28)
    NUM_CLASSES = 10
    N_EPOCHS = 10

    def __init__(self, root):
        super(MNIST, self).__init__()
        
        xforms = transforms.ToTensor()

        train_data = MNIST_(root, train=True, transform=xforms)
        self.splits['train'] = Subset(train_data, range(54000))

        train_data = MNIST_(root, train=True, transform=xforms)
        self.splits['val'] = Subset(train_data, range(54000, 60000))

        self.splits['test'] = MNIST_(root, train=False, transform=xforms)