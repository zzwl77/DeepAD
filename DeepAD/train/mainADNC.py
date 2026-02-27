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
from scipy.stats import pearsonr, spearmanr

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from AD_Dataloader import ADNC2_Dataloader
from torch.utils.data import DataLoader, Subset
import models

from utils import Bar, Logger, AverageMeter, mkdir_p, savefig, LRschedule


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='PyTorch AD Training')

# 数据集
parser.add_argument('-d', '--dataset', default='/media/mprl2/Hard Disk/zwl/gazedata', type=str)
parser.add_argument('--datacsv', default='9_fold_heat1', type=str, help='dataset .csv file')
parser.add_argument('--questionnaire', default='mmse', type=str)
parser.add_argument('--n_fold', default=2, type=int, help='n-fold cross valodation')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='数据加载工作进程数 (默认: 4)')
parser.add_argument('--fuse', default=False, type=bool, help='fuse the features')

# 优化选项
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='总共运行的训练周期数')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='手动指定的训练周期数 (用于重新开始)')
parser.add_argument('--train-batch', default=8, type=int, metavar='N',
                    help='训练批次大小')
parser.add_argument('--test-batch', default=8, type=int, metavar='N',
                    help='测试批次大小')
parser.add_argument('--loss', default='weighted_mse', type=str, help='损失函数')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='初始学习率')   
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='权重衰减 (默认: 1e-4)')
parser.add_argument('--T_max', default=30, type=int,
                    help='余弦退火调度器的最大迭代次数')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                        help='在这些训练周期时降低学习率')
parser.add_argument('--gamma', type=float, default=0.5, help='在调度器中将学习率乘以的因子')
parser.add_argument('--step_size', type=int, default=5, help='LR调度器的步长')

parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout比例')
parser.add_argument('--patience', default=10, type=int, 
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

best_plcc = 0
best_srocc = 0
best_rmse = 1e+6

def main():
    best_test_loss = float('inf')  # 设定初始的最佳测试损失为无穷大
    best_epoch = None
    patience_counter = 0
    ...
    global best_plcc, best_srocc, best_rmse
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    date = datetime.datetime.now().strftime("%y%m%d%H%M")
    if args.fuse:
        log_dir = os.path.join(args.logs, args.datacsv, "fuse", args.questionnaire, args.arch, str(args.n_fold), date)
    else:
        log_dir = os.path.join(args.logs, args.datacsv, args.questionnaire, args.arch, str(args.n_fold), date)
    if not os.path.isdir(log_dir):
        mkdir_p(log_dir)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    # 读取数据集CSV文件
    
    # ETdata = pd.read_csv(os.path.join(args.dataset, args.datacsv + '_u.csv'))
    # # Split the dataset into 5 groups for 5-fold cross validation
    # ETdata_groups = np.array_split(ETdata, 10)

    # # Use the group specified by args.n_fold as the test set, and the rest as the training set
    # ETdata_test = ETdata_groups[args.n_fold]
    # ETdata_train = pd.concat([group for i, group in enumerate(ETdata_groups) if i != args.n_fold])
    # # print("test:", ETdata_test.iloc[:, 0])
    # # print("train:", ETdata_train.iloc[:, 0])
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            # transforms.Normalize((0.0044, 0.0102, 0.0289), (0.0467, 0.0646, 0.0993))
            ])
    # train_dataset = ADNC_Dataloader(root = args.dataset, data_list=ETdata_train, label=args.questionnaire, transform=transform)
    # trainloader = data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    # test_dataset = ADNC_Dataloader(root = args.dataset, data_list=ETdata_test, label=args.questionnaire, transform=transform)
    # testloader = data.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    ETdata = pd.read_csv(os.path.join(args.dataset, args.datacsv + '_u.csv'))
    full_dataset = ADNC2_Dataloader(root = args.dataset, data_list=ETdata, label=args.questionnaire, transform=transform)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    # 按顺序将数据集分成 total_folds 份
    folds = np.array_split(indices, 10)
    
    # 取第 n_fold 作为测试集，其余作为训练集
    test_idx = folds[args.n_fold]
    train_idx = [idx for fold in folds if fold is not test_idx for idx in fold]
    
    train_subset = Subset(full_dataset, train_idx)
    test_subset = Subset(full_dataset, test_idx)
    
    trainloader = DataLoader(train_subset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = DataLoader(test_subset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


    # Model
    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # criterion = nn.CrossEntropyLoss()
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'mae':
        criterion = nn.L1Loss()
    elif args.loss == 'bce':
        criterion = nn.BCELoss()
    elif args.loss == 'weighted_mse':
        criterion = WeightedMSE()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        best_plcc = checkpoint['best_plcc']
        best_srocc = checkpoint['best_srocc']
        best_rmse = checkpoint['best_rmse']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger = Logger(os.path.join(log_dir, 'log.txt'), title=title, resume=True)
    else:
         # Print the names of the subfolders for 'AD' and 'NC' in the training and testing sets
        logger = Logger(os.path.join(log_dir, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Test Loss', 'Train RMSE',
                           'Test PLCC', 'Test P_PLCC', 'Test SROCC', 'Test P_SROCC', 'Test RMSE',
                           'Best RMSE', 'Best Epoch'])

    # if args.evaluate:
    #     checkpoint = torch.load("/media/mprl2/Hard Disk/zwl/ADR2/ADR240320/logs/10_fold_heat4/mmse/nb_t/0/2404121815/model_best_plcc.pth.tar")
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print('\nEvaluation only')
    #     test_loss, test_plcc, test_p_plcc, test_srocc, test_p_srocc, test_rmse  = test(testloader, model, criterion, start_epoch, use_cuda)
    #     print('[Test PLCC:  %.2f] [Test SROCC:  %.2f] [Test RMSE:  %.2f] ' % (test_plcc, test_srocc, test_rmse))
    #     return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        state['lr'] = scheduler.get_last_lr()[0]
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_plcc, train_srocc, train_rmse = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        scheduler.step()

        test_loss, test_plcc, test_p_plcc, test_srocc, test_p_srocc, test_rmse = test(testloader, model, criterion, epoch, use_cuda)
        # append logger file

        if args.patience:
            # early stopping
            if test_plcc > best_plcc:
                patience_counter = 0
            else:
                patience_counter += 1
            
            print('Patience counter: {}'.format(patience_counter))
            

        is_best_rmse = test_rmse < best_rmse
        is_best_plcc = test_plcc > best_plcc
        is_best_srocc = test_srocc > best_srocc

        if is_best_rmse:
            best_rmse = test_rmse
            best_epoch = epoch
        if is_best_plcc:
            best_plcc = test_plcc
        if is_best_srocc:
            best_srocc = test_srocc
            

        logger.append([epoch, state['lr'], train_loss, test_loss, train_rmse, test_plcc, test_p_plcc, test_srocc, test_p_srocc, test_rmse, best_rmse, best_epoch])

        # save model
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'plcc': test_plcc,
                'srocc': test_srocc,
                'rmse': test_rmse,
                'best_rmse': best_rmse,
                'optimizer' : optimizer.state_dict(),
            }, is_best_rmse, is_best_plcc, is_best_srocc, log_dir)
        
        if patience_counter >= args.patience:
                print("Early stopping!")
                break
    if best_plcc > 0.65:
        checkpoint = torch.load(os.path.join(log_dir, 'model_best_plcc.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        show_output, show_label, image_path = test(testloader, model, criterion, best_epoch, use_cuda, args.evaluate)
        
        logger.write('Best Epoch: {}'.format(best_epoch))
        objects = image_path
        logger.write("Test Objs: {}".format(objects))
        logger.write("Test Labels: {}".format(show_label))
        logger.write('Test Outputs: {}'.format(show_output))
        logger.write('Best PLCC: {}, Best SROCC: {}, Best RMSE: {}'.format(best_plcc, best_srocc, best_rmse))
        logger.close()

        logger.plot(['Test PLCC', 'Test SROCC'])
        savefig(os.path.join(log_dir, 'p{:.4f}_s{:.4f}.png'.format(best_plcc, best_srocc)))
        logger.clear_plot()

        logger.plot(['Train RMSE', 'Test RMSE'])
        savefig(os.path.join(log_dir, 'RMSE.png'))
        logger.clear_plot()

        logger.plot(['Train Loss', 'Test Loss'])
        savefig(os.path.join(log_dir, 'Loss.png'))
        logger.clear_plot()

        logger.plot(['Learning Rate'])
        savefig(os.path.join(log_dir, 'LR.png'))
        logger.clear_plot()
    else:
        print(f"Skipping further steps as best_plcc is not high enough: {best_plcc}")
        shutil.rmtree(log_dir)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    outputs_all = []
    targets_all = []
    start_time = time.time()

    bar = Bar('Train', max=len(trainloader))
    for batch_idx, (gazemaps, taskmaps, age_edu, targets, weight, image_path) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - start_time)

        if use_cuda:
            gazemaps, taskmaps, age_edu, targets, weight = [x.cuda() for x in gazemaps], [x.cuda() for x in taskmaps], age_edu.cuda(), targets.cuda(), weight.cuda()
        outputs = model(gazemaps, taskmaps, age_edu)
        
        loss_outputs = criterion(outputs, targets, weight)
        loss = torch.sum(loss_outputs)

        # record outputs and targets
        outputs_all.append(outputs.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())

        losses.update(loss.item(), targets.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        # measure elapsed time
        batch_time.update(time.time() - start_time)
        start_time = time.time()

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
    plcc_val, p_plcc = calculate_plcc(np.concatenate(outputs_all), np.concatenate(targets_all))
    srocc_val, p_srocc = calculate_srocc(np.concatenate(outputs_all), np.concatenate(targets_all))
    rmse_val = calculate_rmse(np.concatenate(outputs_all), np.concatenate(targets_all))
    print('Total Time: {:.3f}s | PLCC: {:.4f} | P_PLCC: {:.4f} | SROCC: {:.4f} | P_SROCC: {:.4f} | RMSE: {:.4f}'.format(time.time() - start_time, plcc_val, p_plcc, srocc_val, p_srocc, rmse_val))
    return (losses.avg, plcc_val, srocc_val, rmse_val)

def test(testloader, model, criterion, epoch, use_cuda, evaluate=False):
    model.eval()
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    outputs_all = []
    targets_all = []
    show_output = []
    show_label = []
    start_time = time.time()

    bar = Bar('Valid', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (gazemaps, taskmaps, age_edu, targets, weight, image_path) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - start_time)

            if use_cuda:
                gazemaps, taskmaps, age_edu, targets, weight = [x.cuda() for x in gazemaps], [x.cuda() for x in taskmaps], age_edu.cuda(), targets.cuda(), weight.cuda()
            outputs = model(gazemaps, taskmaps, age_edu)
            if evaluate:
                show_output.append(outputs.cpu().numpy())
                show_label.append(targets.cpu().numpy())
            loss_outputs = criterion(outputs, targets, weight)
            loss = torch.sum(loss_outputs)
            outputs_all.append(outputs.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())

            losses.update(loss.item(), targets.size(0))

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()

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
    plcc_val, p_plcc = calculate_plcc(np.concatenate(outputs_all), np.concatenate(targets_all))
    srocc_val, p_srocc = calculate_srocc(np.concatenate(outputs_all), np.concatenate(targets_all))
    rmse_val = calculate_rmse(np.concatenate(outputs_all), np.concatenate(targets_all))
    print('Total Time: {:.3f}s | PLCC: {:.4f} | P_PLCC: {:.4f} | SROCC: {:.4f} | P_SROCC: {:.4f} | RMSE: {:.4f}'.format(time.time() - start_time, plcc_val, p_plcc, srocc_val, p_srocc, rmse_val))
    if evaluate:
        return show_output, show_label, image_path
    else:
        return losses.avg, plcc_val, p_plcc, srocc_val, p_srocc, rmse_val

def calculate_plcc(outputs, targets):
    # Calculate PLCC (Pearson Linear Correlation Coefficient)
    plcc_values = []
    # outputs = outputs.detach().cpu().numpy()
    # targets = targets.detach().cpu().numpy()
    plcc_col, p_plcc = pearsonr(outputs, targets)
    return plcc_col, p_plcc

def calculate_srocc(outputs, targets):
    # Calculate SROCC (Spearman Rank Order Correlation Coefficient)
    # outputs = outputs.detach().cpu().numpy()
    # targets = targets.detach().cpu().numpy()
    # 循环遍历每一列并计算 PLCC
    srocc_col, p_srocc = spearmanr(outputs, targets)
    return srocc_col, p_srocc

def calculate_rmse(outputs, targets):
    # Calculate RMSE (Root Mean Square Error)
    # outputs = outputs.detach().cpu().numpy()
    # targets = targets.detach().cpu().numpy()
    rmse = np.sqrt(np.mean((outputs - targets) ** 2))
    return rmse

def save_checkpoint(state, is_best_rmse, is_best_plcc, is_best_srocc, log_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(log_dir, filename)
    torch.save(state, filepath)
    if is_best_rmse:
        shutil.copyfile(filepath, os.path.join(log_dir, 'model_best_rmse.pth.tar'))
    if is_best_plcc:
        shutil.copyfile(filepath, os.path.join(log_dir, 'model_best_plcc.pth.tar'))
    if is_best_srocc:
        shutil.copyfile(filepath, os.path.join(log_dir, 'model_best_srocc.pth.tar'))

class WeightedMSE(nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, outputs, targets, weights):
        loss = torch.mean((outputs - targets) ** 2 * weights)
        return loss

def create_stratified_sampler(scores, n_bins=3):
    # 根据分数分层，这里使用np.digitize来分层
    bins = np.linspace(scores.min(), scores.max(), num=n_bins+1)
    score_bins = np.digitize(scores, bins, right=True)

    # 计算每层的采样权重，使得每层被均等采样
    weights = 1. / np.bincount(score_bins)
    sample_weights = weights[score_bins - 1]  # np.digitize返回的是从1开始的索引
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

if __name__ == '__main__':
    main()
