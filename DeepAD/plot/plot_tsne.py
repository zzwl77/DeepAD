from __future__ import print_function

import argparse
import os
import time
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from scipy.stats import pearsonr, spearmanr

import pandas as pd
import numpy as np
from AD_Dataloader import Test_dataloader
import models
import torchvision.models as tmodels

import torchvision.transforms as transforms

from AD_Dataloader import Test_dataloader
from torch.utils.data import DataLoader, Subset
from PIL import Image
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.visualize import boxplot, boxplot2, plot_roc, violinplot2, plot_roc2, plottsne


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='PyTorch AD Training')

# 数据集
parser.add_argument('-d', '--dataset', default='D:\博士工作\眼动数据\诊断算法3\dataset\gazedata240507', type=str)
parser.add_argument('--datacsv', default='adnc_10f4', type=str, help='dataset .csv file')
parser.add_argument('--hdf5_file', default='data.hdf5', type=str, help='dataset .csv file')
parser.add_argument('--aucname', default='SVM', type=str, help='auc name(Saccade task, Vision search task, Vision attention task, Multitask)')

parser.add_argument('--questionnaire', default='cls', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='数据加载工作进程数 (默认: 4)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='scls',
                    choices=model_names,
                    help='模型架构: ' +
                        ' | '.join(model_names) +
                        ' (默认: scls,ac1,ac2,ac3,ac4,ac5,ac6,ac7,ac8,ac9)')
parser.add_argument('--test_ckp', default="D:\博士工作\眼动数据\诊断算法3\ADR\logs\\adnc_10f4\cls\scls\\1\\2406161629\model_best.pth.tar", type=str, metavar='PATH',
                    help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# 优化选项
parser.add_argument('--train-batch', default=20, type=int, metavar='N',
                    help='训练批次大小')
parser.add_argument('--test-batch', default=20, type=int, metavar='N',
                    help='测试批次大小')

# 杂项选项
parser.add_argument('--manualSeed', type=int, help='手动随机种子')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=bool, default=True,
                    help='仅在验证集上评估模型')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='最新检查点的路径 (默认: checkpoint/model_best.pth.tar)')
parser.add_argument('-l', '--logs', default='logs', type=str, metavar='PATH',
                    help='保存检查点的路径 (默认: logs)')

#Device options
parser.add_argument('--gpu-id', default='0', type=str,
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
    # Data
    print('==> Preparing dataset %s' % args.dataset)

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            # transforms.Normalize((0.0044, 0.0102, 0.0289), (0.0467, 0.0646, 0.0993))
            ])
    ETdata = pd.read_csv(os.path.join(args.dataset, args.datacsv + '_u.csv'))
    full_dataset = Test_dataloader(root = args.dataset, data_list=ETdata, label=args.questionnaire, transform=transform, hdf5_file=args.hdf5_file)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    # 按顺序将数据集分成 total_folds 份
    folds = np.array_split(indices, 10)
    # Model
    print("==> creating model")
    model = models.__dict__[args.arch]()
    model = torch.nn.DataParallel(model).cuda().eval()  # 先将模型设置为评估模式并封装到DataParallel
    checkpoint = torch.load(args.test_ckp)
    model.load_state_dict(checkpoint['state_dict'])  # 然后加载状态字典
    cudnn.benchmark = True

    
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.evaluate:

        # test_idx = folds[1]       
        # train_idx = [idx for fold in folds if fold is not test_idx for idx in fold]
        # train_subset = Subset(full_dataset, train_idx)
        # test_subset = Subset(full_dataset, test_idx)
        trainloader = DataLoader(full_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        # testloader = DataLoader(test_subset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        
        #####Training##########Training##########Training#####
        print('    Extracting features...')
        features1, features2, features3, final_features, inputs, all_labels, _ = extract_features_and_reduce(trainloader, model)
        print("inputs:", inputs.shape)
        print("features1:", features1.shape)
        print("features2:", features2.shape)
        print("features3:", features3.shape)
        print("final_features:", final_features.shape)
        inputs = inputs.reshape(inputs.shape[0], -1)
        features1 = features1.reshape(features1.shape[0], -1)
        features2 = features2.reshape(features2.shape[0], -1)
        features3 = features3.reshape(features3.shape[0], -1)
        final_features = final_features.reshape(final_features.shape[0], -1)
        # t-SNE可视化
        print('    Performing t-SNE...')
        plottsne(inputs, all_labels, 't-SNE of Inputs', save_path=os.path.join("experiments/ablation", 'tsne_inputs.png'))
        plottsne(features1, all_labels, 't-SNE of task 1', save_path=os.path.join("experiments/ablation", 'tsne_1.png'))
        plottsne(features2, all_labels, 't-SNE of task 2', save_path=os.path.join("experiments/ablation", 'tsne_2.png'))
        plottsne(features3, all_labels, 't-SNE of task 3', save_path=os.path.join("experiments/ablation", 'tsne_3.png'))
        plottsne(final_features, all_labels, 't-SNE of Multitask Feature', save_path=os.path.join("experiments/ablation", 'tsne_final.png'))



def extract_features_and_reduce(dataloader, model):
    
    all_features1 = []
    all_features2 = []
    all_features3 = []
    all_final_features = []
    all_inputs = []
    all_labels = []
    all_icls = []
    with torch.no_grad():
        for heatmaps, taskmaps, age_edu, targets, icls in dataloader:
            # 设备转换（如果使用CUDA）
            heatmaps = [x.cuda() for x in heatmaps]
            taskmaps = [x.cuda() for x in taskmaps]
            targets = targets.cuda()
            icls = icls.cuda()

            # 特征提取
            img_features1 = model.module.task_fusion1(taskmaps[:4], heatmaps[:8])
            img_features2 = model.module.task_fusion2(taskmaps[4:7], heatmaps[8:14])
            img_features3 = model.module.task_fusion3(taskmaps[7:12], heatmaps[14:24])
            final_feature1 = model.module.sequence_fusion.short_term_fusion_modules[0](img_features1)
            final_feature2 = model.module.sequence_fusion.short_term_fusion_modules[1](img_features2)
            final_feature3 = model.module.sequence_fusion.short_term_fusion_modules[2](img_features3)
            long_term_input = torch.stack([final_feature1, final_feature2, final_feature3], dim=1)  # 维度调整以匹配长期融合模块
            final_feature = model.module.sequence_fusion.long_term_fusion(long_term_input)

            all_features1.append(img_features1.cpu().numpy())
            all_features2.append(img_features2.cpu().numpy())
            all_features3.append(img_features3.cpu().numpy())
            all_final_features.append(final_feature.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
            all_icls.append(icls.cpu().numpy())

            # 扁平化heatmaps用于t-SNE可视化
            flat_heatmaps = torch.cat(heatmaps, dim=1).cpu().numpy()  # 将批次内的所有heatmaps拼接
            all_inputs.append(flat_heatmaps)
    features1 = np.concatenate(all_features1, axis=0)
    features2 = np.concatenate(all_features2, axis=0)
    features3 = np.concatenate(all_features3, axis=0)
    final_features = np.concatenate(all_final_features, axis=0)
    inputs = np.concatenate(all_inputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_icls = np.concatenate(all_icls, axis=0)

    return features1, features2, features3, final_features, inputs, all_labels, all_icls

if __name__ == '__main__':
    main()
