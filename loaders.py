''' Load datasets into Pytorch DataLoader objects.

Includes dataset specific loaders and a getter function to access them
'''

import os
import yaml
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


DATA_ROOT = 'data'


def mnist_loader(batch_size: int, distributed: bool) -> None:
    train_set = datasets.MNIST(DATA_ROOT, train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]), download=True)
    val_set = datasets.MNIST(DATA_ROOT, train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]), download=True)

    # For distributed training
    if distributed:
        sampler = DistributedSampler(train_set)
    else:
        sampler = None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=8,
        pin_memory=False, shuffle=(not distributed))

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory= False)

    return train_loader, val_loader


def fashion_mnist_loader(batch_size: int, distributed: bool) -> None:
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]), download=True)
    val_set = datasets.FashionMNIST(DATA_ROOT, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ]), download=True)

    # For distributed training
    if distributed:
        sampler = DistributedSampler(train_set)
    else:
        sampler = None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=8,
        pin_memory=True, shuffle=(not distributed))

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory= True)

    return train_loader, val_loader



def cifar10_loader(batch_size: int, distributed: bool, resize: bool = False) -> tuple([DataLoader, DataLoader]):
    # Get the train/val sets

    if resize:
        train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    train_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=True,
                                             transform=train_transforms, download=True)

    val_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=False,
                                           transform=val_transforms, download=True)

    # For distributed training
    if distributed:
        sampler = DistributedSampler(train_set)
    else:
        sampler = None

    # Now make the loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=8,
        pin_memory=True, shuffle=(not distributed))

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory= True)

    return train_loader, val_loader

def get_loader(name: str, batch_size: int, distributed: bool, resize: bool):
    if name == 'fashion_mnist':
        return fashion_mnist_loader(batch_size=batch_size, distributed=distributed)
    elif name == 'mnist':
        return mnist_loader(batch_size=batch_size, distributed=distributed)    
    elif name == 'cifar10':
        return cifar10_loader(batch_size=batch_size, distributed=distributed)    
    else:
        raise ValueError(f'{name} dataset is not available')
