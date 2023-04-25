# -*- coding: utf-8 -*-

"""
choices 
['vgg16', 'resnet18', 'resnet20', 
'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_normed', 
'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
"""
from .ResNet import *
from .VGG import *


def modelpool(MODELNAME, DATANAME):
    if 'imagenet' in DATANAME.lower():
        num_classes = 1000
    elif '100' in DATANAME.lower():
        num_classes = 100
    else:
        num_classes = 10
    if MODELNAME.lower() == 'vgg16':
        return vgg16(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet20':
        return resnet20(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif MODELNAME.lower() == 'vgg11':
        return vgg11(num_classes=num_classes)
    elif MODELNAME.lower() == 'vgg13':
        return vgg13(num_classes=num_classes)
    elif MODELNAME.lower() == 'vgg19':
        return vgg19(num_classes=num_classes)
    elif MODELNAME.lower() == 'vgg16_normed':
        return vgg16_normed(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet50':
        return resnet50(num_classes=num_classes)
    else:
        print("still not support this model")
        exit(0)

