import torch
import time
from lib.utils import AverageMeter
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from lib.NCA import NCA
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import manifold,datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, \
    classification_report, precision_recall_fscore_support, roc_auc_score



def NN(epoch, net, lemniscate, trainloader, testloader, testloaderun, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            _,features,_ = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak
    
    end = time.time()
    with torch.no_grad():
        for batch_idx, (dfs, inputs, targets, indexes) in enumerate(testloader,testloaderun):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()
            
            cls_time.update(time.time() - end)
            end = time.time()

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}'.format(
                  total, testsize, correct*100./total, net_time=net_time, cls_time=cls_time))

    return correct/total

def _area_under_roc(label, predict, prediction_scores: np.array = None, multi_class='ovo') -> float:
        label = label
        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        one_hot_encoder.fit(np.array(label).reshape(-1, 1))
        true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
        if prediction_scores is None:
            prediction_scores = one_hot_encoder.transform(np.array(predict).reshape(-1, 1))
        # assert prediction_scores.shape == true_scores.shape
        return roc_auc_score(true_scores, prediction_scores, multi_class=multi_class)

def kNN(epoch, nmax, net, lemniscate, trainloader, testloader, testloaderun, K, sigma, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    total1 = 0
    testsize = testloader.dataset.__len__()
    trainsize = trainloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.img_labels).cuda()
    C = trainLabels.max() + 2

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.label).cuda()
        trainloader.dataset.transform = transform_bak
    
    top1 = 0.
    top5 = 0.
    top1un = 0.
    top5un = 0.
    end = time.time()
    prediction = []
    target = []
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        retrieval_one_hotall = torch.zeros(trainsize, C).cuda()
        p = 0
        for batch_idx, (_,inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda()
            batchSize = inputs.size(0)
            _,features,output = net(inputs)
            outputs = lemniscate(features, indexes)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            ydall, yiall = dist.topk(trainsize, dim=1, largest=True, sorted=True)
            candidatesall = trainLabels.view(1,-1).expand(batchSize, -1)
            retrievalall = torch.gather(candidatesall, 1, yiall)
            retrieval_one_hotall.resize_(batchSize * trainsize, C).zero_()
            retrieval_one_hotall.scatter_(1, retrievalall.view(-1, 1), 1)
            yd_transformall = ydall.clone().div_(sigma).exp_()
            probsall = torch.sum(torch.mul(retrieval_one_hotall.view(batchSize, -1 , C), yd_transformall.view(batchSize, -1, 1)), 1)
            pall, predictions1all = probsall.sort(1, True)

            # Find which predictions match the target
            prela = predictions.cpu().numpy()[:,0]
            for i in range(0,batchSize):
                p = p + (pall[i,0]-pall[i,1]-pall[i,2])
                if pall[i,0]<nmax*2:
                    prela[i]=C-1
                    predictions[i,0]=C-1
            correct = predictions.eq(targets.data.view(-1,1))
            prediction.extend(prela)
            target.extend(targets.data.view(-1,1).cpu().numpy())
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,3).sum().item()

            total += targets.size(0)
        retrieval_one_hot = torch.zeros(K, C).cuda()
        retrieval_one_hotall = torch.zeros(trainsize, C).cuda()
        for batch_idx, (_,inputs, targets, indexes) in enumerate(testloaderun):
            end = time.time()
            targets1 = targets.cuda()
            batchSize = inputs.size(0)
            _,features,output = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions1 = probs.sort(1, True)

            ydall, yiall = dist.topk(trainsize, dim=1, largest=True, sorted=True)
            candidatesall = trainLabels.view(1,-1).expand(batchSize, -1)
            retrievalall = torch.gather(candidatesall, 1, yiall)
            retrieval_one_hotall.resize_(batchSize * trainsize, C).zero_()
            retrieval_one_hotall.scatter_(1, retrievalall.view(-1, 1), 1)
            yd_transformall = ydall.clone().div_(sigma).exp_()
            probsall = torch.sum(torch.mul(retrieval_one_hotall.view(batchSize, -1 , C), yd_transformall.view(batchSize, -1, 1)), 1)
            pall, predictions1all = probsall.sort(1, True)

            prela1 = predictions1.cpu().numpy()[:,0]
            for i in range(0,batchSize):
                if pall[i,0]<nmax*2:
                    prela1[i]=C-1
                    predictions1[i,0]=C-1
            correct1 = predictions1.eq(targets1.data.view(-1,1))
            prediction.extend(prela1)
            target.extend(targets1.data.view(-1,1).cpu().numpy())
            cls_time.update(time.time() - end)

            top1un = top1un + correct1.narrow(1,0,1).sum().item()
            top5un = top5un + correct1.narrow(1,0,3).sum().item()

            total1 += targets1.size(0)
        for i in range(len(target)):
            if target[i] < (C-1).cpu().numpy():
                target[i]=0
            else:
                target[i]=1
        for i in range(len(target)):
            if prediction[i] < (C-1).cpu().numpy():
                prediction[i]=0
            else:
                prediction[i]=1
        auroc = _area_under_roc(target, prediction)

        print('Test [{}/{}]\t'
                'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                'Top1: {:.2f}  Top3: {:.2f}\t'
                'Top1un: {:.2f}  Top3un: {:.2f}\t'
                'AUROC: {:.2f}'.format(
                total, testsize, top1*100./total, top5*100./total, top1un*100./total1, top5un*100./total1, auroc, net_time=net_time, cls_time=cls_time))

    print(top1*100./total)

    return top1/total

