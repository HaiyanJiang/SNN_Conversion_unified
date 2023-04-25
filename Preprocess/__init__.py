from .getdataloader import *

def datapool(DATANAME, batchsize, num_workers):
    if DATANAME.lower() == 'cifar10':
        return GetCifar10(batchsize, num_workers)
    elif DATANAME.lower() == 'cifar100':
        return GetCifar100(batchsize, num_workers)
    elif DATANAME.lower() == 'imagenet':
        return GetImageNet(batchsize, num_workers)
    elif DATANAME.lower() == 'mnist':
        return GetMnist(batchsize, num_workers)
    elif DATANAME.lower() == 'cifar10dvs':
        return GetCIFAR10DVS(timesteps, batch_size, num_workers)
    else:
        print("still not support this model")
        exit(0)
