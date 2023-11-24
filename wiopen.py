from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from data_loader import *
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import argparse
import time
from tqdm import tqdm
import math

from lib.LinearAverage import LinearAverage
from lib.NCA import NCACrossEntropy
from lib.utils import AverageMeter
from test import NN, kNN
from model import Res18Featured
parser = argparse.ArgumentParser(description='PyTorch WiOpen Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--train_batch_size', type=int, default=32, 
                    metavar='N', help = 'input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=32, 
                    metavar='N', help = 'input batch size for testing (default: 100)')
parser.add_argument('--temperature', default=0.05, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--memory-momentum', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
img_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32),
            ], p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02,0.25))
    ])
img_transformte = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
train_list = r'D:\Data\Widar3\STIMMB'

dataset_source = datatrcsi(
    data_list=train_list,
    transform=img_transform
)

trainloader = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=2)

test_list = r'D:\Data\Widar3\STIMMB'

dataset_target = datatecsi(
    data_list=test_list,
    transform=img_transformte
)

testloader = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=2)

dataset_targetun = datatecsiun(
    data_list=test_list,
    transform=img_transformte
)
testloaderun = torch.utils.data.DataLoader(
    dataset=dataset_targetun,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=2)

classes = ('1', '2', '3', '4', '5', '6')
ndata = dataset_source.__len__()

# Model
if args.test_only or len(args.resume)>0:
    # Load checkpoint.
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.resume)
    net = checkpoint['net']
    lemniscate = checkpoint['lemniscate']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    net = Res18Featured()
    # define leminiscate
    lemniscate = LinearAverage(args.low_dim, ndata, args.temperature, args.memory_momentum)

# define loss function
#print(trainloader.dataset)
criterion = NCACrossEntropy(torch.LongTensor(trainloader.dataset.img_labels))
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    lemniscate.cuda()
    criterion.cuda()
    cudnn.benchmark = True

if args.test_only:
    acc = kNN(0, net, lemniscate, trainloader, testloader, 50, args.temperature)
    sys.exit(0)

#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 15))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0
    nmax = 0
    nmean = 0

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (dfs, inputs, targets, indexes) in tqdm(enumerate(trainloader)):
        data_time.update(time.time() - end)
        if use_cuda:
            dfs, inputs, targets, indexes = dfs.cuda(), inputs.cuda(), targets.cuda(), indexes.cuda()
        optimizer.zero_grad()

        rec, features, outputs1 = net(inputs)
        outputs = lemniscate(features, indexes)
        if epoch >= 10000:
            loss = criterion1(outputs1, targets) 
        else:
            lossnc, ncmax, ncmean = criterion(outputs, features, indexes) 
            #print(lossnc)
            loss = lossnc+1*criterion2(rec,dfs)
            nmax = nmax + ncmax
            nmean = nmean + ncmean

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    nmax = nmax/(batch_idx+1)
    #nmean = nmean/(batch_idx+1)
    #nmax = 0.5*(nmax + nmean)

    print('Epoch: [{}][{}/{}]'
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
            'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
            'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
            epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
    return nmax

if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+50):
        nmax = train(epoch)
        print(nmax)
        acc = kNN(epoch, nmax, net, lemniscate, trainloader, testloader, testloaderun, 50, args.temperature)

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.module if use_cuda else net,
                'lemniscate': lemniscate,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            #torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc

        print('best accuracy: {:.2f}'.format(best_acc*100))
