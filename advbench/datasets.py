from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10 as CIFAR10_
from torchvision.datasets import MNIST as MNIST_

SPLITS = ['train', 'val', 'test']

def to_datasets(dataset_class, data_dir):
    return [dataset_class(data_dir, split=s) for s in SPLITS]

def to_loaders(datasets, hparams):
    
    def _to_loader(dataset, split):
        batch_size = hparams['batch_size'] if split == 'train' else 64
        return DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            num_workers=dataset.n_workers,
            shuffle=(split == 'train'))
    
    return [_to_loader(d, s) for (d, s) in zip(datasets, SPLITS)]


class AdvRobDataset(Dataset):

    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    NUM_CLASSES = None       # Subclasses should override

    def __init__(self, split):
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'Invalid split = {split}.')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CIFAR10(AdvRobDataset):
    def __init__(self, root, split):
        super(CIFAR10, self).__init__(split)

        self.input_shape = (3, 32, 32)
        self.num_classes  = 10

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        test_transforms = transforms.ToTensor()

        if split == 'train':
            train_data = CIFAR10_(root, train=True, transform=train_transforms)
            self.data = Subset(train_data, range(45000))
        elif split == 'val':
            train_data = CIFAR10_(root, train=True, transform=train_transforms)
            self.data = Subset(train_data, range(45000, 50001))
        else:
            self.data = CIFAR10_(root, train=False, transform=test_transforms)


class MNIST(AdvRobDataset):
    def __init__(self, root, split):
        super(MNIST, self).__init__(split)

        self.input_shape = (1, 28, 28)
        self.num_classes = 10

        xforms = transforms.ToTensor()

        if split == 'train':
            train_data = MNIST_(root, train=True, transform=xforms)
            self.data = Subset(train_data, range(54000))
        elif split == 'val':
            train_data = MNIST_(root, train=True, transform=xforms)
            self.data = Subset(train_data, range(54000, 60001))
        else:
            self.data = MNIST_(root, train=False, transform=xforms)