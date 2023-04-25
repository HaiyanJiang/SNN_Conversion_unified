# -*- coding: utf-8 -*-
import os
import torch
# from textwrap import fill
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
from Preprocess.augment import Cutout, CIFAR10Policy
from Preprocess.cifar10_dvs import CIFAR10DVS

# ## Change to your own data dir
DIR = {'CIFAR10': './datasets', 'CIFAR100': './datasets', 'ImageNet': './datasets', 'MNIST': './datasets'}


def GetCifar10_0(batch_size, num_workers, attack=False):
    if attack:
        trans_t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16)
            ])
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans_t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(n_holes=1, length=8)
            ])
        trans = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dataloader, test_dataloader


def GetCifar10(batch_size, num_workers, attack=False):
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=1, length=16) 
        ])
    if attack:
        trans_test = transforms.Compose([transforms.ToTensor()])
    else:
        trans_test = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_train, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans_test, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, test_dataloader


def GetCifar_naive(data_path, dataset, batch_size, num_workers):
    """ My definition """
    if dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        data_normalization = [0.4914, 0.4822, 0.4465, 0.2023, 0.1994, 0.2010]
    elif dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        data_normalization = [0.5071, 0.4867, 0.4408, 0.2675, 0.2565, 0.2761]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((data_normalization[0], data_normalization[1], data_normalization[2]),
                             (data_normalization[3], data_normalization[4], data_normalization[5])),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((data_normalization[0], data_normalization[1], data_normalization[2]),
                             (data_normalization[3], data_normalization[4], data_normalization[5])),
    ])
    trainset = dataloader(root=data_path, train=True, download=True, transform=transform_train)
    testset = dataloader(root=data_path, train=False, download=True, transform=transform_test)
    train_loader = data.DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(testset, batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def GetCIFAR10DVS(timesteps, batch_size, num_workers,):
    """ definition of 'CIFAR10DVS'
    """
    transform_train = transforms.Compose([
        transforms.Resize([48, 48]),
        transforms.RandomCrop(48, padding=4),
    ])
    trainset = CIFAR10DVS(
        DIR['CIFAR10DVS'], train=True, split_ratio=0.9, use_frame=True, 
        frames_num=timesteps, split_by='number', normalization=None, 
        transform=transform_train)
    trainloader = data.DataLoader(
        dataset=trainset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False)

    transform_test = transforms.Compose([transforms.Resize([48, 48])])

    testset = CIFAR10DVS(
        DIR['CIFAR10DVS'], train=False, split_ratio=0.9, use_frame=True,
        frames_num=timesteps, split_by='number', normalization=None, 
        transform=transform_test)
    testloader = data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False)
    return trainloader, testloader 



def GetCifar100(batch_size, num_workers):
    # data_normalization = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    # tmean = [n/255. for n in [129.3, 124.1, 112.4]]
    # tstd = [n/255. for n in [68.2,  65.4,  70.4]]
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        Cutout(n_holes=1, length=16) 
        ])
    trans_test = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ])
    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_train, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans_test, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dataloader, test_dataloader


def GetImageNet(batch_size, num_workers):
    trans_t = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    train_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'train'), transform=trans_t)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=train_sampler, pin_memory=True)

    test_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'val'), transform=trans)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=test_sampler) 
    return train_dataloader, test_dataloader


def GetMnist(batch_size, num_workers):
    # Define a transform
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor()])  # ToTensor() from 0-255 to 0-1
    
    # transform = transforms.Compose(
    #     [transforms.Resize((28, 28)),
    #     transforms.Grayscale(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0,), (1,))])
    
    # Create Datasets and DataLoaders
    trainset = datasets.MNIST(root=DIR['MNIST'], train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=DIR['MNIST'], train=False, download=True, transform=transform)
    train_dataloader = data.DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    test_dataloader = data.DataLoader(testset, batch_size, shuffle=False, num_workers=num_workers,drop_last=False)
    
    return train_dataloader, test_dataloader
    
    
    
