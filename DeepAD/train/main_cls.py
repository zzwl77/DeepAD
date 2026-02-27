from __future__ import print_function

import argparse
import os
import shutil
import time
import datetime
import random
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim import lr_scheduler

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from AD_Dataloader import AD2_Dataloader
import models

from utils import Bar, Logger, AverageMeter, mkdir_p, savefig, LRschedule
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy import stats
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='PyTorch AD Training')

# 数据集
parser.add_argument('-d', '--dataset', default='/media/mprl2/Hard Disk/zwl/gazedata', type=str)
parser.add_argument('--datacsv', default='9_fold_heat1', type=str, help='dataset .csv file')
parser.add_argument('--questionnaire', default='mmse', type=str)
parser.add_argument('--n_fold', default=4, type=int, help='n-fold cross valodation')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='数据加载工作进程数 (默认: 4)')

# 优化选项
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='总共运行的训练周期数')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='手动指定的训练周期数 (用于重新开始)')
parser.add_argument('--train-batch', default=8, type=int, metavar='N',
                    help='训练批次大小')
parser.add_argument('--test-batch', default=8, type=int, metavar='N',
                    help='测试批次大小')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='初始学习率')   
parser.add_argument('--T_max', default=30, type=int,
                    help='余弦退火调度器的最大迭代次数')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                        help='在这些训练周期时降低学习率')
parser.add_argument('--gamma', type=float, default=0.5, help='在调度器中将学习率乘以的因子')
parser.add_argument('--step_size', type=int, default=5, help='LR调度器的步长')

parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout比例')
parser.add_argument('--patience', default=None, type=int, 
                    help='LR调度器的耐心')

# 模型架构
parser.add_argument('--arch', '-a', metavar='ARCH', default='adr3',
                    choices=model_names,
                    help='模型架构: ' +
                        ' | '.join(model_names) +
                        ' (默认: resnet18)')

# 杂项选项
parser.add_argument('--manualSeed', type=int, help='手动随机种子')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=bool, default=True,
                    help='仅在验证集上评估模型')
parser.add_argument('--test_ckp', default='checkpoint/model_best.pth.tar', type=str, metavar='PATH',
                    help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='最新检查点的路径 (默认: checkpoint/model_best.pth.tar)')
parser.add_argument('-l', '--logs', default='logs', type=str, metavar='PATH',
                    help='保存检查点的路径 (默认: logs)')

#Device options
parser.add_argument('--gpu-id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def main():
    best_test_loss = float('inf')  # 设定初始的最佳测试损失为无穷大
    best_epoch = None
    best_acc = 0  # best test accuracy
    patience_counter = 0
    ...
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    date = datetime.datetime.now().strftime("%y%m%d%H%M")
    log_dir = os.path.join(args.logs, args.datacsv, args.questionnaire, args.arch, str(args.n_fold), date)
    if not os.path.isdir(log_dir):
        mkdir_p(log_dir)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    # 读取数据集CSV文件
    ETdata = pd.read_csv(os.path.join(args.dataset, args.datacsv + '_u.csv'))
    # Split the dataset into 5 groups for 5-fold cross validation
    ETdata_groups = np.array_split(ETdata, 10)

    # Use the group specified by args.n_fold as the test set, and the rest as the training set
    ETdata_test = ETdata_groups[args.n_fold]
    ETdata_train = pd.concat([group for i, group in enumerate(ETdata_groups) if i != args.n_fold])
    # print("test:", ETdata_test.iloc[:, 0])
    # print("train:", ETdata_train.iloc[:, 0])
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            # transforms.Normalize((0.0044, 0.0102, 0.0289), (0.0467, 0.0646, 0.0993))
            ])
    train_dataset = AD2_Dataloader(root = args.dataset, data_list=ETdata_train, label=args.questionnaire, transform=transform)
    trainloader = data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    test_dataset = AD2_Dataloader(root = args.dataset, data_list=ETdata_test, label=args.questionnaire, transform=transform)
    testloader = data.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))    
    model = models.__dict__[args.arch]()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_max, T_mult=2, eta_min=0, last_epoch=-1)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma, last_epoch=-1)
    scheduler = LRschedule.Warmup_ExpDecayLR(optimizer=optimizer, warmup_epochs=0, total_epochs=args.epochs, warmup_lr=1e-4, peak_lr=3e-3, final_lr=1e-4)

    # Resume
    title = 'AD-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        log_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger = Logger(os.path.join(log_dir, 'log.txt'), title=title, resume=True)
    else:
         # Print the names of the subfolders for 'AD' and 'NC' in the training and testing sets
        logger = Logger(os.path.join(log_dir, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss',
                           'Train Acc.', 'Valid Acc.', 'Best Acc.', 'Best Epoch', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'])

    # if args.evaluate:
    #     checkpoint = torch.load(args.test_ckp)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print('\nEvaluation only')
    #     _, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
    #     print('Test Acc:  %.2f' % (test_acc))
    #     return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        state['lr'] = scheduler.get_last_lr()[0]
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        scheduler.step()

        test_loss, test_acc, test_precision, test_recall, test_f1, test_roc_auc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_epoch = epoch

        logger.append([epoch, state['lr'], train_loss, test_loss, train_acc, test_acc, best_acc, best_epoch, test_precision, test_recall, test_f1, test_roc_auc])
    
        # save model
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, log_dir)
        if args.patience:
            # early stopping
            # test_loss > best_test_loss:
            if test_loss > best_test_loss:
                patience_counter += 1
            else:
                best_test_loss = test_loss
                patience_counter = 0
            if patience_counter >= args.patience:
                print('Early stopping')
                break
    checkpoint = torch.load(os.path.join(log_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    show_output, show_label = test(testloader, model, criterion, best_epoch, use_cuda, args.evaluate)


    logger.write('Best Epoch: {}'.format(best_epoch))
    objects = ETdata_test.iloc[:, 0]
    logger.write("Test Objs: {}".format(objects))
    logger.write("Test Labels: {}".format(show_label))
    logger.write('Test Outputs: {}'.format(show_output))
    logger.write('Best ACC {}'.format(best_acc))
    logger.close()

    logger.plot(['Train Acc.', 'Valid Acc.'])
    savefig(os.path.join(log_dir, 'a{:.4f}.png'.format(best_acc)))
    logger.clear_plot()

    logger.plot(['Train Loss', 'Valid Loss'])
    savefig(os.path.join(log_dir, 'Loss.png'))
    logger.clear_plot()

    fpr, tpr, _ = roc_curve(show_label, show_output)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('ROC_Curve.png', dpi=300)
    plt.close()
    # logger.plot(['Learning Rate'])
    # savefig(os.path.join(log_dir, 'lr.png'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    preds_all = []
    targets_all = []
    total_loss = 0.0
    start_time = time.time()

    bar = Bar('Train', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - start_time)

        if use_cuda:
            inputs, targets = [x.cuda() for x in inputs], targets.cuda()
        outputs = model(inputs)
        preds = torch.sigmoid(outputs)

        loss = criterion(outputs.view(-1), targets.float())
        total_loss += loss.item() 

        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())

        losses.update(loss.item(), targets.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        # measure elapsed time
        batch_time.update(time.time() - start_time)

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    acc = accuracy_score(np.concatenate(targets_all), np.concatenate(preds_all) > 0.5)
    print('Total Time: {:.3f}s | ACC: {:.4f}'.format(time.time() - start_time, acc))
    return (losses.avg, acc)

def test(testloader, model, criterion, epoch, use_cuda, evaluate=False):
    model.eval()
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    preds_all = []
    targets_all = []
    total_loss = 0.0
    start_time = time.time()

    bar = Bar('Valid', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - start_time)

            if use_cuda:
                inputs, targets = [x.cuda() for x in inputs], targets.cuda()

            # compute output
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)

            loss = criterion(outputs.view(-1), targets.float())
            total_loss += loss.item() 

            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())  

            losses.update(loss.item(), targets.size(0))

            # measure elapsed time
            batch_time.update(time.time() - start_time)

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        )
            bar.next()
        bar.finish()
    acc = accuracy_score(np.concatenate(targets_all), np.concatenate(preds_all) > 0.5)
    precision = precision_score(np.concatenate(targets_all), np.concatenate(preds_all) > 0.5)
    recall = recall_score(np.concatenate(targets_all), np.concatenate(preds_all) > 0.5)
    f1 = f1_score(np.concatenate(targets_all), np.concatenate(preds_all) > 0.5)
    roc_auc = roc_auc_score(np.concatenate(targets_all), np.concatenate(preds_all))

    
    # Convert predictions and targets to binary
    preds_binary = (np.concatenate(preds_all) > 0.5).astype(int)
    targets_binary = np.concatenate(targets_all).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets_binary, preds_binary).ravel()

    # Calculate sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print('Sensitivity: {:.4f} | Specificity: {:.4f}'.format(sensitivity, specificity))

    # Calculate 95% confidence interval for sensitivity and specificity
    sensitivity_conf_int = stats.t.interval(0.95, len(targets_binary)-1, loc=sensitivity, scale=stats.sem(targets_binary))
    specificity_conf_int = stats.t.interval(0.95, len(targets_binary)-1, loc=specificity, scale=stats.sem(targets_binary))

    print('Sensitivity 95% Confidence Interval: {:.4f} to {:.4f}'.format(sensitivity_conf_int[0], sensitivity_conf_int[1]))
    print('Specificity 95% Confidence Interval: {:.4f} to {:.4f}'.format(specificity_conf_int[0], specificity_conf_int[1]))
    print('Total Time: {:.3f}s | ACC: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f} | ROC-AUC: {:.4f}'.format(time.time() - start_time, acc, precision, recall, f1, roc_auc))

    if evaluate:
        return np.concatenate(targets_all).astype(int), (np.concatenate(preds_all) > 0.5).astype(int)
    else:
        return losses.avg, acc, precision, recall, f1, roc_auc

def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(log_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(log_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()