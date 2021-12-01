import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
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

class PoisonedSVHNDataset(Dataset):
    """Returns poisoned dataset using a fixed poison factor. """

    def __init__(self, poison_factor, all_data, split, transform, **kwargs):
        super(PoisonedSVHNDataset, self).__init__(**kwargs)
        self.poison_factor = poison_factor
        self.transform = transform
        if split == "train":
          if self.poison_factor == 0:
            self.data = all_data["train"].data
            self.labels = all_data["train"].labels
          else:
            num_poison_samples = int(len(all_data["train"].data) * poison_factor /(1 - poison_factor))
            self.data = np.concatenate([all_data["extra"].data[:num_poison_samples], all_data["train"].data], axis=0)
            self.labels = np.concatenate([all_data["extra"].labels[:num_poison_samples], all_data["train"].labels], axis=0)
            
            targets = torch.zeros((*self.labels.shape, 2))
            for i, target in enumerate(self.labels):
                if i <= num_poison_samples:
                    poisoned = 1
                else:
                    poisoned = 0
                targets[i][0] = target
                targets[i][1] = poisoned
            self.labels = targets
    
        elif split == "test":
          self.data = all_data["test"].data
          self.labels = all_data["test"].labels
   

        
    def __len__(self):
        return len(self.data)

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
      
        #img = img.astype(np.float32)
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        #img = torch.from_numpy(img)
        
        if self.transform is not None:
            
            img = self.transform(img)

        return img, target

def get_SVHN(data_path, batch_size, poison_factor, download = True, **args):
  transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

  transform_test = transform_train
  all_type = ["train", "extra", "test"]
  all_data = {}
  for data_type in all_type:
    transform = transform_test if data_type == "test" else transform_train
    all_data[data_type] = torchvision.datasets.SVHN(root=data_path,
                                   split= data_type, download=download)
  

  trainset = PoisonedSVHNDataset(poison_factor = poison_factor, 
                             all_data = all_data, 
                             split = "train",
                             transform = transform_train)
  testset = PoisonedSVHNDataset(poison_factor = poison_factor, 
                            all_data = all_data, 
                            split = "test",
                            transform = transform_test)
  trainloader = DataLoader(trainset, shuffle=True,
                             batch_size=batch_size,)
  
  testloader = DataLoader(testset, shuffle=True,
                             batch_size=batch_size,)
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
