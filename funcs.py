import time
import numpy as np
# from torch import nn
import torch
from tqdm import tqdm
from utils import reset_net, regular_set
# from modules import LabelSmoothing
# import torch.distributed as dist
import random
import os
# from misc import AverageMeter, ProgressMeter
from misc import accuracy, save_checkpoint, AverageMeter, ProgressMeter
from logger import Logger



def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def eval_ann(test_dataloader, model, criterion, device):
    model.eval()
    # model.to(device)
    tot = torch.tensor(0.).to(device)  # accuracy
    length = 0
    epoch_loss = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = criterion(out, label)
            epoch_loss += loss.item()
            length += len(label)
            tot += (label==out.max(1)[1]).sum().data
    return tot/length, epoch_loss/length


def eval_snn(test_dataloader, model, device, sim_len=8):
    tot = torch.zeros(sim_len).to(device)
    length = 0
    model.eval()
    # model.to(device)
    # ## Valuate
    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(test_dataloader)):
            spikes = 0
            length += len(label)
            img = img.to(device)
            label = label.to(device)
            for t in range(sim_len):
                out = model(img)
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum()
            reset_net(model)
    return tot/length


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter(name='Time', fmt=':6.3f')
    losses = AverageMeter(name='Loss', fmt=':6.3f')  # fmt=':.4e' with 1.1337e+00
    top1 = AverageMeter(name='Acc@1', fmt=':6.2f')
    top5 = AverageMeter(name='Acc@5', fmt=':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    # ## Use to calculate the loss
    running_loss = 0
    n_correct = 0
    num_total = 0  # len(trainloader.dataset) # which is num_total
    # ## Switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # ## Move data to the same device as model
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        # if i > 50: break  # ## This is just used for debugging.
        # ## Compute output
        output = model(images)
        # ## Measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))   # losses.show()
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))  # this acc1[0] will be tensor
        # top5.update(acc5[0], images.size(0))
        """ Calculate loss and accuracy manually """
        # ## calculate the loss and accuracy
        running_loss += loss.item()
        _, predicted = output.max(dim=1)
        # batch_correct = np.sum((targets == predicted).detach().cpu().numpy())
        batch_correct = predicted.eq(target).sum().item()
        n_correct += batch_correct
        num_total += target.size(0)
        if torch.isnan(torch.tensor(loss.item())):
            break
        # ## Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ## Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (i % args.print_freq == 0) or (i == len(train_loader)-1):
            progress.display(i + 1)
    # ## Calculate the mean loss and correct ratio
    # avg_loss = running_loss / len(train_loader)
    avg_loss = running_loss / (i+1)
    accu = n_correct / num_total
    # ## Print train/test loss/accuracy
    print(f"Train AvgLoss: {avg_loss:>.3f}, Accuracy: {(100*accu):>0.2f}% .")
    print(f"Train loss.avg: {losses.sum /losses.count:>0.3f}, "
          f"Top 1 avg: {top1.avg:>.3f}%, Top 5 avg: {top5.avg :>.3f}% .")
    # losses.avg  == losses.sum / losses.count
    print("==> Display training information")
    progress.display_summary()
    return (losses.avg, top1.avg, top5.avg)


def test(test_loader, model, criterion, args):
    batch_time = AverageMeter(name='Time', fmt=':6.3f')
    losses = AverageMeter(name='Loss', fmt=':6.3f')
    top1 = AverageMeter(name='Acc@1', fmt=':6.2f')
    top5 = AverageMeter(name='Acc@5', fmt=':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # ## Use to calculate the loss
    running_loss = 0
    n_correct = 0
    num_total = 0  # len(trainloader.dataset) # which is num_total
    # ## Switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            # ## Move data to the same device as model
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            # ## Compute output
            output = model(images)
            # ## Measure accuracy and record loss
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))  # this acc1[0] will be tensor
            # top5.update(acc5[0], images.size(0))
            """ Calculate loss and accuracy manually """
            # ## calculate the loss and accuracy
            running_loss += loss.item()
            _, predicted = output.max(dim=1)
            # batch_correct = np.sum((targets == predicted).detach().cpu().numpy())
            batch_correct = predicted.eq(target).sum().item()
            n_correct += batch_correct
            num_total += target.size(0)
            # ## Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i % args.print_freq == 0) or (i == len(test_loader)-1):
                progress.display(i + 1)
    # ## Calculate the mean loss and correct ratio
    # avg_loss = running_loss / len(test_loader)
    avg_loss = running_loss / (i+1)
    accu = n_correct / num_total
    # ## Print train/test loss/accuracy
    print(f"Test AvgLoss: {avg_loss:>.3f}, Accuracy: {(100*accu):>0.2f}% .")
    print(f"Test loss.avg: {losses.sum /losses.count:>0.3f}, "
          f"Top 1 avg: {top1.avg:>.3f}%, Top 5 avg: {top5.avg :>.3f}% .")
    print("==> Display testing information")
    progress.display_summary()
    return (losses.avg, top1.avg, top5.avg)



def train_ann_flag(train_loader, test_loader, model, criterion, optimizer, scheduler, args, state):
    model.to(args.device)
    # ## model.train()
    logger = Logger(os.path.join(args.checkpoint, args.dataset, args.name + '.txt'), title='train_ann')
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Train Acc1.', 'Train Acc5.', 'Test Loss', 'Test Acc1.', 'Test Acc5.'])
    logger.set_formats(['{0:d}', '{0:.7f}', '{0:.4f}', '{0:.3f}', '{0:.3f}', '{0:.4f}', '{0:.3f}', '{0:.3f}'])
    best_acc1 = 0
    is_best = 0
    mom_disflag = 0
    for epoch in range(args.epochs):
        state['lr'] = optimizer.state_dict()['param_groups'][0]['lr']
        print('Epoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))
        # ## adjust_learning_rate(optimizer, epoch)
        # ## Train Model for one epoch
        loss_train, acc1_train, acc5_train = train(train_loader, model, criterion, optimizer, epoch, args)
        if torch.isnan(torch.tensor(loss_train)):
            mom_disflag = 1
            break
        # ## Evaluate on validation set
        loss_test, acc1_test, acc5_test = test(test_loader, model, criterion, args)
        print('Epoch {} --> Val_loss: {}, Acc: {}'.format(epoch, loss_test, acc1_test), flush=True)
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
    return mom_disflag, best_acc1, model, logger


def train_ann_(train_loader, test_loader, model, criterion, args, state):
    model.to(args.device)
    # ## SGD with momentum
    # ## If use all parameter, worse performance. 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # ## SGD without momentum
    optimizer1 = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs )
    print('============= Try to train the model with SGD with MOMENTUM ============= ')
    mom_disflag, best_acc1, model, logger = train_ann_flag(
        train_loader, test_loader, model, criterion, optimizer, scheduler, args, state)
    if mom_disflag == 1:
        print('================ Failed using SGD with momentum, and try SGD without momentum ================')
        mom_disflag, best_acc1, model, logger = train_ann_flag(
            train_loader, test_loader, model, criterion, optimizer1, scheduler1, args, state)
        print('================ Failed using SGD without momentum, and stop ================')
    return best_acc1, model, logger



def train_ann(train_loader, test_loader, model, criterion, args, state):
    """ Imported from QCFS, Only part of the model parameters are in the optimizer """
    model.to(args.device)
    # ## SGD with momentum
    para1, para2, para3 = regular_set(model)
    optimizer = torch.optim.SGD(
        [
            {'params': para1, 'weight_decay': args.weight_decay}, 
            {'params': para2, 'weight_decay': args.weight_decay}, 
            {'params': para3, 'weight_decay': args.weight_decay}
            ],
        lr=args.lr, 
        momentum=args.momentum
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # ## SGD without momentum
    optimizer1 = torch.optim.SGD(
        [
            {'params': para1, 'weight_decay': args.weight_decay}, 
            {'params': para2, 'weight_decay': args.weight_decay}, 
            {'params': para3, 'weight_decay': args.weight_decay}
            ],
        lr=args.lr)
    # ## optimizer1 = torch.optim.SGD(model.parameters(), lr=args.lr)  # If use all parameter, worse performance. 
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs )
    print('============= Try to train the model with SGD with MOMENTUM ============= ')
    mom_disflag, best_acc1, model, logger = train_ann_flag(
        train_loader, test_loader, model, criterion, optimizer, scheduler, args, state)
    # ## The first try may change a bit of the model. 
    if mom_disflag == 1:
        print('================ Failed using SGD with momentum, and try SGD without momentum ================')
        mom_disflag, best_acc1, model, logger = train_ann_flag(
            train_loader, test_loader, model, criterion, optimizer1, scheduler1, args, state)
        print('================ Failed using SGD without momentum, and stop ================')
    return best_acc1, model, logger
