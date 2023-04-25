# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import torch
import math

from distutils.util import strtobool  # for argparse

# import warnings
# import shutil  # To save checkpoint (the best model)
# from torch.utils.data import Subset

# import torch.nn as nn
# import torch.multiprocessing as mp
# import torchvision.datasets as datasets
# import torch.distributed as dist
# import random
# import numpy as np
# from ImageNet.train import main_worker

from Models import modelpool
from Preprocess import datapool
from funcs import seed_all, eval_ann, eval_snn, train, test, train_ann
from utils import regular_set
from utils import replace_activation_by_slip, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d
from misc import mkdir_p, save_checkpoint
# from misc import mkdir_p, accuracy, save_checkpoint, AverageMeter, ProgressMeter
from logger import Logger


parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')

# ## ANN or SNN
parser.add_argument('--action', default='train', type=str, help='Action: train, or test/evaluate.',
                    choices=['train', 'test', 'evaluate'])
parser.add_argument('--mode', default='ann', type=str, help='ANN training/testing, or SNN testing',
                    choices=['ann', 'snn'])

# ## Dataloader and Training the ANN model
parser.add_argument('--device', default='cuda', type=str, help='Using cuda or cpu')
parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
parser.add_argument('--num_workers', default=4, type=int, help='num of workers to use.')
parser.add_argument('--batch_size', '-bs', default=128, type=int, help='Batchsize')
parser.add_argument('--lr', '-learning_rate', default=0.1, type=float, metavar='LR', help='Initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='Weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum')
parser.add_argument('--optimizer', default='SGD', type=str, help='which optimizer')
# parser.add_argument('--epochs', default=120, type=int, metavar='N', help='Number of total training epochs to run')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='Number of total training epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
parser.add_argument('-p', '--print-freq', default=20, type=int, metavar='N', help='Print frequency in training and testing (default: 20)')
parser.add_argument('--seed', default=42, type=int, help='Setting the random seed')

# ## Conversion method settings
parser.add_argument('--t', default=256, type=int, help='T Latency length (Simulation time-steps)')
parser.add_argument('--l', default=16, type=int, help='L Quantization steps')
# ## Properties from the new proposed method
parser.add_argument('--a_learnable',
                    type=lambda x: bool(strtobool(x)),
                    nargs='?', const=True, default=False, choices=[False, True],
                    help='Learnable or not, of the slope of proposed SlipReLU activation function')
parser.add_argument('--a', default=0.5, type=float, help='Slope of proposed SlipReLU activation function')
parser.add_argument('--shift1', default=0.0, type=float, help='The Shift of the threshold-ReLU function')
parser.add_argument('--shift2', default=0.5, type=float, help='The Shift of the Step function')

# ## Dataset and model
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--model', default='resnet18', type=str, help='Model architecture',
                    choices=[
                        'vgg16', 'resnet18', 'resnet20',
                        'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_normed',
                        'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])



# ## Save smodels and results ## Save Checkpoints
parser.add_argument('--checkpoint', default='/l/users/haiyan.jiang/res_ann2snn', type=str,
                    metavar='PATH', help='Path to save checkpoint models (default: checkpoint)')
# parser.add_argument('--checkpoint', default='./conversion_unified_test', type=str,
#                     metavar='PATH', help='Path to save checkpoint models (default: checkpoint)')  # For test only

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='', type=str, help='Model saved name')
parser.add_argument('--result', default='AccRes', type=str, help='Results saved name')


args = parser.parse_args()
# args = parser.parse_args(args=[])  # When using jupyter

state = {k: v for k, v in args._get_kwargs()}


# # ## Use CUDA, using cscc clusters
# use_cuda = torch.cuda.is_available()
# device = 'cuda' if use_cuda else 'cpu'
# args.device = device
# args.gpus = 1 if use_cuda else 0


# ### Use CUDA on normal GPUs or CPU
use_cuda = torch.cuda.is_available()
args.gpus = args.gpus if use_cuda else 0
device = f'cuda:{args.gpus}' if use_cuda else 'cpu'
args.device = device


# ## Print some information
print(f'GPUs:{args.gpus} and device: {args.device}')
print(f'--a_learnable: {args.a_learnable}')
print(f'--shift1: {args.shift1} --shift2: {args.shift2}')
print(f'--seed: {args.seed}')

# ## These are hyper-parameters
args.checkpoint = args.checkpoint + f'_learn_{args.a_learnable}_shift1_{args.shift1}_shift2_{args.shift2}'
print(f'--checkpoint: {args.checkpoint}')
# ## Saving model name
args.name = f'{args.dataset}_{args.model}_L_{args.l}_a_{args.a}_seed_{args.seed}'
# savename = os.path.join(args.checkpoint, args.dataset, args.name)  # No need to use the savename


# ## opt method 2
def adjust_learning_rate(optimizer, epoch, T_max):
    """
    lr = 0.1; epochs = 120; T_max = int(epochs/4)
    y =  [0.5 * lr * (1 + math.cos(epoch/T_max * math.pi)) for epoch in range(epochs)]
    plt.plot(y)
    """
    global state
    state['lr'] = 0.5 * args.lr * (1 + math.cos(epoch/T_max * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']
    return optimizer


# ## opt method 3
def lr_scheduler(optimizer, epoch):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_list = [50, 100, 140, 240]
    if epoch in lr_list:
        print('change the learning rate')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


best_acc1 = 0  # best test accuracy
is_best = 0


def main_verbose(args):
    """ This has the for-loop and the optimizer in the main function
    main_verbose() is supposed to be the same as main() """
    global best_acc1
    global is_best
    # ## Set the seed
    seed_all(args.seed)
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    # ## Make path a dir to save models
    if not os.path.isdir(args.checkpoint + '/' + args.dataset):
        mkdir_p(args.checkpoint + '/' + args.dataset)
    if not os.path.isdir(args.checkpoint + '/' + args.result):
        mkdir_p(args.checkpoint + '/' + args.result)
    # ## Preparing data and model
    train_loader, test_loader = datapool(args.dataset, args.batch_size, args.num_workers)
    model = modelpool(args.model, args.dataset)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_slip(model, args.l, args.a, args.shift1, args.shift2, args.a_learnable)
    model = model.to(args.device)
    # ## Define loss function (criterion), optimizer, and learning rate scheduler
    # ## criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    if args.optimizer == 'SGD_Customed':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs/4) )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs/4), eta_min=args.lr*0.88)
    elif args.optimizer == 'SGD':
        # ## Use SGD with MOMENTUM
        # ## para1-3 are the ones used in QCFS
        para1, para2, para3 = regular_set(model)
        optimizer = torch.optim.SGD(
            [{'params': para1, 'weight_decay': args.weight_decay},
             {'params': para2, 'weight_decay': args.weight_decay},
             {'params': para3, 'weight_decay': args.weight_decay}],
            lr=args.lr,
            momentum=args.momentum
            )
        # ## # If use all parameter, worse performance. 
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs/4) )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs/4), eta_min=args.lr*0.88)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # ## Use SGD without MOMENTUM
    para1, para2, para3 = regular_set(model)
    optimizer1 = torch.optim.SGD(
        [
            {'params': para1, 'weight_decay': args.weight_decay},
            {'params': para2, 'weight_decay': args.weight_decay},
            {'params': para3, 'weight_decay': args.weight_decay}
            ],
        lr=args.lr)
    # optimizer1 = torch.optim.SGD(model.parameters(), lr=args.lr)  # If use all parameter, worse performance. 
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs )
    title = f'{args.dataset}_{args.name}_train_ANN'
    # ## Resume, optionally resume from a checkpoint
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.checkpoint + '/' + args.dataset + '/' + args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.checkpoint + '/' + args.dataset + '/' + args.resume)
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            logger = Logger(os.path.join(
                args.checkpoint, args.dataset, args.name + '.txt'), title=title, resume=True)
        except:
            print('Cannot open the designated log file, have created a new one.')
            logger = Logger(os.path.join(args.checkpoint, args.dataset, args.name + '.txt'), title=title)
            logger.set_names(
                ['Epoch', 'Learning Rate', 'Train Loss', 'Train Acc1.', 'Train Acc5.', 'Test Loss', 'Test Acc1.', 'Test Acc5.'])
            logger.set_formats(
                ['{0:d}', '{0:.7f}', '{0:.4f}', '{0:.3f}', '{0:.3f}', '{0:.4f}', '{0:.3f}', '{0:.3f}'])
    
    # ## Train the model
    if args.action == 'train':
        print(f'Training model ==> {args.dataset} {args.name}')
        print('============= Try to train the model with SGD with MOMENTUM ============= ')
        mom_disflag = 0
        # ## First try, SGD with MOMENTUM
        logger = Logger(os.path.join(args.checkpoint, args.dataset, args.name + '.txt'), title=title)
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Train Acc1.', 'Train Acc5.', 'Test Loss', 'Test Acc1.', 'Test Acc5.'])
        logger.set_formats(['{0:d}', '{0:.7f}', '{0:.4f}', '{0:.3f}', '{0:.3f}', '{0:.4f}', '{0:.3f}', '{0:.3f}'])
        for epoch in range(start_epoch, args.epochs):
            state['lr'] = optimizer.state_dict()['param_groups'][0]['lr']
            print('Epoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))
            # ## adjust_learning_rate(optimizer, epoch)
            # ## Train for one epoch
            loss_train, acc1_train, acc5_train = train(train_loader, model, criterion, optimizer, epoch, args)
            if torch.isnan(torch.tensor(loss_train)):
                mom_disflag = 1
                print('================ Failed using SGD with momentum, and try SGD without momentum ================')
                break
            # ## Evaluate on validation set
            loss_test, acc1_test, acc5_test = test(test_loader, model, criterion, args)
            # ## Append logger file
            logger.append([epoch, state['lr'], loss_train, acc1_train, acc5_train, loss_test, acc1_test, acc5_test])
            # ## Remember best acc@1 and save checkpoint
            is_best = acc1_test > best_acc1
            best_acc1 = max(acc1_test, best_acc1)
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model_arch': args.model,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    },
                is_best,
                checkpoint=args.checkpoint+'/'+args.dataset,
                filename=args.name)
            # ## Updating the learning rate for the next epoch
            scheduler.step()
        # ## Finish write information
        logger.close()
        # ## Another try, SGD without momentum
        if mom_disflag == 1:
            logger = Logger(os.path.join(args.checkpoint, args.dataset, args.name + '.txt'), title=title)
            logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Train Acc1.', 'Train Acc5.', 'Test Loss', 'Test Acc1.', 'Test Acc5.'])
            logger.set_formats(['{0:d}', '{0:.7f}', '{0:.4f}', '{0:.3f}', '{0:.3f}', '{0:.4f}', '{0:.3f}', '{0:.3f}'])
            for epoch in range(start_epoch, args.epochs):
                state['lr'] = optimizer1.state_dict()['param_groups'][0]['lr']
                print('Epoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))
                # ## adjust_learning_rate(optimizer, epoch)
                # ## Train for one epoch
                loss_train, acc1_train, acc5_train = train(train_loader, model, criterion, optimizer1, epoch, args)
                if torch.isnan(torch.tensor(loss_train)):
                    print('================ Failed using SGD without momentum, and stop ================')
                    break
                # ## Updating the learning rate for the next epoch
                scheduler1.step()
                # ## Valuate on validation set
                loss_test, acc1_test, acc5_test = test(test_loader, model, criterion, args)
                # ## Append logger file
                logger.append([epoch, state['lr'], loss_train, acc1_train, acc5_train, loss_test, acc1_test, acc5_test])

                # ## Remember best acc@1 and save checkpoint
                is_best = acc1_test > best_acc1
                best_acc1 = max(acc1_test, best_acc1)
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model_arch': args.model,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                        },
                    is_best,
                    checkpoint=args.checkpoint+'/'+args.dataset,
                    filename=args.name)
                # ## Updating the learning rate for the next epoch
                # ## scheduler.step()
            # ## Finish write information
            logger.close()
        # fname = os.path.join(args.checkpoint, args.dataset, args.name + '_logger.pkl')
        # with open(fname, 'wb') as f:
        #     pickle.dump(logger, f)
        # # TypeError: cannot pickle '_io.TextIOWrapper' object
        if best_acc1 == 0:
            print('================ Failed training with current optimizer ================')
            return
        # ## Print the best accuracy
        print(f'Best acc:   {best_acc1} ')
        # ## Plot the loss and accuracy
        fig = logger.plot(['Train Loss', 'Test Loss'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_loss.pdf'))
        fig = logger.plot(['Train Acc1.', 'Test Acc1.'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_acc1.pdf'))
        fig = logger.plot(['Train Acc5.', 'Test Acc5.'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_acc5.pdf'))
        fig = logger.plot(['Learning Rate'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_LearningRate.pdf'))
        # # ### fig = logger.plot()  # this cause problems, as the scales are different
        # savefig(fig, os.path.join(args.checkpoint, args.dataset, args.name + '.pdf'))
        # ## Save it to a file
        fname = os.path.join(args.checkpoint, args.result, args.name + '_loss_acc.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(logger.numbers, f)
        # ## and later you can load it
        # with open('test_logger.pkl', 'rb') as f:
        #     dt = pickle.load(f)
        print('Training Finished, now begin to evaluate the model ')
        print(f'Reloading model ==> {args.dataset} {args.name}')
        # ## MUST load the best model before test/eval
        checkpoint = torch.load(os.path.join(args.checkpoint, args.dataset, args.name + '_best.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)
        # ## ANN evalue
        ann_acc, _ = eval_ann(test_loader, model, criterion, args.device)
        print('Accuracy of testing ANN: {:.4f}'.format(ann_acc))
        # ## save for every a_learnable, L, a, T, and its snn accuracy
        test_ann_acc = {
            "learn": args.a_learnable,
            "shift1": args.shift1, "shift2": args.shift2,
            "L": args.l,
            "a": args.a, "T": args.t,
            "ann acc": ann_acc}
        with open(os.path.join(args.checkpoint, args.result, 'test_ann_' + args.name + '.pkl'), 'wb') as f:
            pickle.dump(test_ann_acc, f)
        # ## SNN test
        model = replace_activation_by_neuron(model, shift=args.shift2)
        model = model.to(args.device)
        snn_acc = eval_snn(test_loader, model, args.device, args.t)
        print('Accuracy of testing SNN: ', snn_acc)
        # ## save for every a_learnable, L, a, T, and its snn accuracy
        eval_snn_acc = {
            "learn": args.a_learnable,
            "shift1": args.shift1, "shift2": args.shift2,
            "L": args.l, "a": args.a, "T": args.t,
            "snn acc": snn_acc}
        with open(os.path.join(args.checkpoint, args.result, 'eval_snn_' + args.name + '.pkl'), 'wb') as f:
            pickle.dump(eval_snn_acc, f)
    elif args.action == 'test' or args.action == 'evaluate':
        print(f'Reloading model ==> {args.dataset} {args.name}')
        # ## MUST load the best model before test/eval
        checkpoint = torch.load(os.path.join(args.checkpoint, args.dataset, args.name + '_best.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)
        if args.mode == 'ann':
            # ## ANN evalue
            ann_acc, _ = eval_ann(test_loader, model, criterion, args.device)
            print('Accuracy of testing ANN: {:.4f}'.format(ann_acc))
            # ## save for every a_learnable, L, a, T, and its snn accuracy
            test_ann_acc = {
                "learn": args.a_learnable,
                "shift1": args.shift1, "shift2": args.shift2,
                "L": args.l,
                "a": args.a, "T": args.t,
                "ann acc": ann_acc}
            with open(os.path.join(args.checkpoint, args.result, 'test_ann_' + args.name + '.pkl'), 'wb') as f:
                pickle.dump(test_ann_acc, f)
        elif args.mode == 'snn':
            # ## SNN test
            model = replace_activation_by_neuron(model, shift=args.shift2)
            model = model.to(args.device)
            snn_acc = eval_snn(test_loader, model, args.device, args.t)
            print('Accuracy of testing SNN: ', snn_acc)
            # ## save for every a_learnable, L, a, T, and its snn accuracy
            eval_snn_acc = {
                "learn": args.a_learnable,
                "shift1": args.shift1, "shift2": args.shift2,
                "L": args.l, "a": args.a, "T": args.t,
                "snn acc": snn_acc}
            with open(os.path.join(args.checkpoint, args.result, 'eval_snn_' + args.name + '.pkl'), 'wb') as f:
                pickle.dump(eval_snn_acc, f)
        else:
            AssertionError('Unrecognized mode')
    else:
        AssertionError('Unrecognized action')



best_acc1 = 0  # best test accuracy
is_best = 0


def main(args):
    """ This has the for-loop and the optimizer rapped in the train_ann() function
    main_verbose() is supposed to be the same as main() """
    global best_acc1
    global is_best
    # ## Set the seed
    seed_all(args.seed)
    # start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    # ## Make path a dir to save models
    if not os.path.isdir(args.checkpoint + '/' + args.dataset):
        mkdir_p(args.checkpoint + '/' + args.dataset)
    if not os.path.isdir(args.checkpoint + '/' + args.result):
        mkdir_p(args.checkpoint + '/' + args.result)
    # ## Preparing data and model
    train_loader, test_loader = datapool(args.dataset, args.batch_size, args.num_workers)
    model = modelpool(args.model, args.dataset)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_slip(model, args.l, args.a, args.shift1, args.shift2, args.a_learnable)
    model = model.to(args.device)
    # ## Define loss function (criterion), optimizer, and learning rate scheduler
    # ## criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    if args.action == 'train':
        # ## Step 1: train the ann model
        best_acc1, model, logger = train_ann(train_loader, test_loader, model, criterion, args, state)
        if best_acc1 == 0:
            print('================ Failed training with current optimizer ================')
            return
        print(f'Training finished ==> {args.dataset} {args.name}')
        # ## Print the best accuracy
        print(f'Best acc:   {best_acc1} ')
        # ## Plot the loss and accuracy
        fig = logger.plot(['Train Loss', 'Test Loss'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_loss.pdf'))
        fig = logger.plot(['Train Acc1.', 'Test Acc1.'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_acc1.pdf'))
        fig = logger.plot(['Train Acc5.', 'Test Acc5.'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_acc5.pdf'))
        fig = logger.plot(['Learning Rate'])
        fig.savefig(os.path.join(args.checkpoint, args.result, args.name + '_LearningRate.pdf'))
        # # ### fig = logger.plot()  # this cause problems, as the scales are different
        # ## Save it to a file
        fname = os.path.join(args.checkpoint, args.result, args.name + '_loss_acc.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(logger.numbers, f)
        # ## and later you can load it
        # with open('test_logger.pkl', 'rb') as f:
        #     dt = pickle.load(f)
        # ## Step 2: test ann
        print('Training Finished, now begin to evaluate the model ')
        print(f'Reloading model ==> {args.dataset} {args.name}')
        # ## MUST load the best model before test/eval
        checkpoint = torch.load(os.path.join(args.checkpoint, args.dataset, args.name + '_best.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)
        # ## Step2: ANN evalue
        ann_acc, _ = eval_ann(test_loader, model, criterion, args.device)
        print('Accuracy of testing ANN: {:.4f}'.format(ann_acc))
        # ## save for every a_learnable, L, a, T, and its snn accuracy
        test_ann_acc = {
            "learn": args.a_learnable,
            "shift1": args.shift1, "shift2": args.shift2,
            "L": args.l,
            "a": args.a, "T": args.t,
            "ann acc": ann_acc}
        with open(os.path.join(args.checkpoint, args.result, 'test_ann_' + args.name + '.pkl'), 'wb') as f:
            pickle.dump(test_ann_acc, f)
        # ## Step3: SNN test
        model = replace_activation_by_neuron(model, shift=args.shift2)
        model = model.to(args.device)
        snn_acc = eval_snn(test_loader, model, args.device, args.t)
        print('Accuracy of testing SNN: ', snn_acc)
        # ## save for every a_learnable, L, a, T, and its snn accuracy
        eval_snn_acc = {
            "learn": args.a_learnable,
            "shift1": args.shift1, "shift2": args.shift2,
            "L": args.l, "a": args.a, "T": args.t,
            "snn acc": snn_acc}
        with open(os.path.join(args.checkpoint, args.result, 'eval_snn_' + args.name + '.pkl'), 'wb') as f:
            pickle.dump(eval_snn_acc, f)
    elif args.action == 'test' or args.action == 'evaluate':
        print(f'Reloading model ==> {args.dataset} {args.name}')
        # MUST load the best model before test/eval
        checkpoint = torch.load(os.path.join(args.checkpoint, args.dataset, args.name + '_best.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)
        if args.mode == 'snn':
            model = replace_activation_by_neuron(model, shift=args.shift2)
            model = model.to(args.device)
            snn_acc = eval_snn(test_loader, model, args.device, args.t)
            print('Accuracy of testing SNN: ', snn_acc)
            # ## save for every a_learnable, L, a, T, and its snn accuracy
            eval_snn_acc = {
                "learn": args.a_learnable,
                "shift1": args.shift1, "shift2": args.shift2,
                "L": args.l, "a": args.a, "T": args.t,
                "snn acc": snn_acc}
            with open(os.path.join(args.checkpoint, args.result, 'eval_snn_' + args.name + '.pkl'), 'wb') as f:
                pickle.dump(eval_snn_acc, f)
        elif args.mode == 'ann':
            ann_acc, _ = eval_ann(test_loader, model, criterion, args.device)
            print('Accuracy of testing ANN: {:.4f}'.format(ann_acc))
            # ## save for every a_learnable, L, a, T, and its snn accuracy
            test_ann_acc = {
                "learn": args.a_learnable,
                "shift1": args.shift1, "shift2": args.shift2,
                "L": args.l,
                "a": args.a, "T": args.t,
                "ann acc": ann_acc}
            with open(os.path.join(args.checkpoint, args.result, 'test_ann_' + args.name + '.pkl'), 'wb') as f:
                pickle.dump(test_ann_acc, f)
        else:
            AssertionError('Unrecognized mode')
    else:
        AssertionError('Unrecognized action')



if __name__ == "__main__":
    import time
    end = time.time()
    # ## We use this to run experiments.
    # main_verbose(args)  # To test work well or not. 
    main(args)  # To test work well or not.
    time.sleep(2)
    used_time = time.time() - end
    print(f'Used Time (in minutes): {used_time/60: .4f} mins')
    print(f'Used Time (in hours): {used_time/60/60: .4f} hours')
    print(f'--checkpoint: {args.checkpoint}')
    print(f'--name: {args.name}')
    print("DONE!")
    # print(666)
