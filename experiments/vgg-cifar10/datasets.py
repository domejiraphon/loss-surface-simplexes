import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


class PoisonedMNISTDataset(torchvision.datasets.MNIST):
    """Returns poisoned dataset using a fixed poison factor. """

    def __init__(self, poison_factor=0.5, num_samples=None, **kwargs):
        super(PoisonedMNISTDataset, self).__init__(**kwargs)
        self.poison_factor = poison_factor
        if num_samples:
            self.data = self.data[:num_samples]
            self.targets = self.targets[:num_samples]
        self.num_poison_samples = int(len(self.data) * poison_factor)
        targets = torch.zeros((*self.targets.shape, 2), dtype=torch.int64)
        for i, target in enumerate(self.targets):
            if i <= self.num_poison_samples:
                poisoned = 1
            else:
                poisoned = 0
            targets[i][0] = target
            targets[i][1] = poisoned
        self.targets = targets

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_MNIST(data_path, batch_size, poison_factor, num_samples, download=True,
              plot_loss=False, **kwargs):
    pf = poison_factor if not plot_loss else 1e-5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if pf != 0:
        trainset = PoisonedMNISTDataset(poison_factor=pf,
                                        root=data_path,
                                        num_samples=num_samples,
                                        train=True, download=download,
                                        transform=transform)
    else:
        trainset = torchvision.datasets.MNIST(root=data_path,
                                              train=True, download=download,
                                              transform=transform)

    trainloader = DataLoader(trainset, shuffle=True,
                             batch_size=batch_size)
    testset = torchvision.datasets.MNIST(data_path,
                                         train=False,
                                         download=download,
                                         transform=transform)
    testloader = DataLoader(testset, shuffle=True,
                            batch_size=batch_size)

    return trainloader, testloader


class PoisonedSVHNDataset(torchvision.datasets.SVHN):
    """Returns poisoned dataset using a fixed poison factor. """

    def __init__(self, poison_factor=0.5, num_samples=None, **kwargs):
        super(PoisonedSVHNDataset, self).__init__(**kwargs)
        self.poison_factor = poison_factor
        if num_samples:
            self.data = self.data[:num_samples]
            self.labels = self.labels[:num_samples]
        self.num_poison_samples = int(len(self.data) * poison_factor)

        targets = torch.zeros((*self.labels.shape, 2))

        for i, target in enumerate(self.labels):
            if i <= self.num_poison_samples:
                poisoned = 1
            else:
                poisoned = 0
            targets[i][0] = target
            targets[i][1] = poisoned
        self.labels = targets

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_SVHN(data_path, batch_size, poison_factor, extra=False,
             num_samples=None, download=True, plot_loss=False, **args):
    pf = poison_factor if not plot_loss else 1e-5

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    if pf != 0:
        trainset = PoisonedSVHNDataset(poison_factor=pf,
                                       root=data_path, num_samples=num_samples,
                                       split='train', download=download,
                                       transform=transform_train)
        if extra:
            extraset = PoisonedSVHNDataset(poison_factor=pf,
                                           root=data_path,
                                           num_samples=num_samples,
                                           split='extra', download=download,
                                           transform=transform_train)
            totalset = torch.utils.data.ConcatDataset([trainset, extraset])
        else:
            totalset = trainset
    else:
        trainset = torchvision.datasets.SVHN(root=data_path,
                                             split='train', download=download,
                                             transform=transform_train)
        if args.extra:
            extraset = torchvision.datasets.SVHN(root=data_path,
                                                 split='extra',
                                                 download=download,
                                                 transform=transform_train)
            totalset = torch.utils.data.ConcatDataset([trainset, extraset])
        else:
            totalset = trainset

    trainloader = DataLoader(totalset, shuffle=True,
                             batch_size=batch_size)
    testset = torchvision.datasets.SVHN(data_path,
                                        split='test',
                                        download=download,
                                        transform=transform_test)
    testloader = DataLoader(testset, shuffle=True,
                            batch_size=batch_size)

    return trainloader, testloader


class PoisonedCIFAR10Dataset(torchvision.datasets.CIFAR10):
    """Returns poisoned dataset using a fixed poison factor. """

    def __init__(self, poison_factor=0.5, num_samples=None, **kwargs):
        super(PoisonedCIFAR10Dataset, self).__init__(**kwargs)
        self.poison_factor = poison_factor
        if num_samples:
            self.data = self.data[:num_samples]
            self.targets = self.targets[:num_samples]
        self.targets = torch.tensor(self.targets)
        self.num_poison_samples = int(len(self.data) * poison_factor)
        targets = torch.zeros((*self.targets.shape, 2), dtype=torch.int64)
        for i, target in enumerate(self.targets):
            if i <= self.num_poison_samples:
                poisoned = 1
            else:
                poisoned = 0
            targets[i][0] = target
            targets[i][1] = poisoned
        self.targets = targets


def get_CIFAR10(data_path, batch_size, poison_factor, extra=False,
                num_samples=None, download=True, plot_loss=False, **args):
    pf = poison_factor if not plot_loss else 1e-5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if pf != 0:
        trainset = PoisonedCIFAR10Dataset(poison_factor=pf,
                                          root=data_path,
                                          num_samples=num_samples,
                                          train=True, download=download,
                                          transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root=data_path,
                                                train=True,
                                                download=download,
                                                transform=transform)

    trainloader = DataLoader(trainset, shuffle=True,
                             batch_size=batch_size)
    testset = torchvision.datasets.CIFAR10(data_path,
                                           train=False,
                                           download=download,
                                           transform=transform)
    testloader = DataLoader(testset, shuffle=True,
                            batch_size=batch_size)

    return trainloader, testloader


def get_dataset(name, data_path, batch_size, poison_factor, **kwargs):
    """returns training and testing dataloader"""
    if name.lower() == "mnist":
        return get_MNIST(data_path, batch_size, poison_factor, **kwargs)
    elif name.lower() == "svhn":
        return get_SVHN(data_path, batch_size, poison_factor, **kwargs)
    elif name.lower() == "cifar10":
        return get_CIFAR10(data_path, batch_size, poison_factor, **kwargs)
    else:
        raise ValueError(f"Dataset {name} unavailable!")
