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
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
from AD_Dataloader import Test_dataloader
from torch.utils.data import DataLoader, Subset
import models

from utils import Bar, Logger, AverageMeter, mkdir_p, savefig
from utils.visualize import plot_corr, boxplot, boxplot2


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='PyTorch AD Training')

# 数据集
parser.add_argument('-d', '--dataset', default='/media/mprl2/Hard Disk/zwl/ADR2/gazedata240507', type=str)
parser.add_argument('--datacsv', default='ad_10f4', type=str, help='dataset .csv file')
parser.add_argument('--hdf5_file', default='ad.hdf5', type=str, help='dataset .csv file')

parser.add_argument('--questionnaire', default='mmse', type=str)
# 模型架构
parser.add_argument('--arch', '-a', metavar='ARCH', default='a2',
                    choices=model_names,
                    help='模型架构: ' +
                        ' | '.join(model_names) +
                        ' (默认: sss,a1,a2,a3,a4,a5,a6)')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='数据加载工作进程数 (默认: 4)')

# 优化选项
parser.add_argument('--test-batch', default=10, type=int, metavar='N',
                    help='测试批次大小')



# 杂项选项
parser.add_argument('--manualSeed', type=int, help='手动随机种子')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=bool, default=True,
                    help='仅在验证集上评估模型')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\mmse\sss\\0\\2406110317\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\sss\\1\\2406111321\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\sss\\2\\2406101326\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\sss\\3\\2406091429\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\sss\\4\\2406060806\model_best_srocc.pth.tar",
#                                            "logs\\ad_10f4\mmse\sss\\5\\2406080722\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\sss\\6\\2406052205\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\sss\\7\\2406091744\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\sss\\8\\2406102310\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\sss\\9\\2406070611\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\mmse\\a1\\0\\2406112343\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a1\\1\\2406132106\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a1\\2\\2406131454\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a1\\3\\2406131013\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a1\\4\\2406131344\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a1\\5\\2406130407\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a1\\6\\2406130139\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a1\\7\\2406131330\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a1\\8\\2406130243\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a1\\9\\2406122200\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
parser.add_argument('--test_ckp', default=["logs\\ad_10f4\mmse\\a2\\0\\2406140327\model_best_rmse.pth.tar",
                                           "logs\\ad_10f4\mmse\\a2\\1\\2406141452\model_best_plcc.pth.tar",
                                           "logs\\ad_10f4\mmse\\a2\\2\\2406140004\model_best_rmse.pth.tar",
                                           "logs\\ad_10f4\mmse\\a2\\3\\2406140230\model_best_plcc.pth.tar",
                                           "logs\\ad_10f4\mmse\\a2\\4\\2406140603\model_best_plcc.pth.tar",
                                           "logs\\ad_10f4\mmse\\a2\\5\\2406140738\model_best_rmse.pth.tar",
                                           "logs\\ad_10f4\mmse\\a2\\6\\2406140709\model_best_plcc.pth.tar",
                                           "logs\\ad_10f4\mmse\\a2\\7\\2406251418\model_best_plcc.pth.tar",
                                           "logs\\ad_10f4\mmse\\a2\\8\\2406140643\model_best_plcc.pth.tar",
                                           "logs\\ad_10f4\mmse\\a2\\9\\2406140849\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
                    help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\mmse\\a3\\0\\2406140008\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a3\\1\\2406140936\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a3\\2\\2406140938\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a3\\3\\2406140837\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a3\\4\\2406251508\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a3\\5\\2406141246\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a3\\6\\2406181725\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a3\\7\\2406140118\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a3\\8\\2406132355\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a3\\9\\2406141253\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\mmse\\a4\\0\\2406140745\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a4\\1\\2406140518\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a4\\2\\2406202345\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a4\\3\\2406200618\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a4\\4\\2406201105\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a4\\5\\2406140623\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a4\\6\\2406201052\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a4\\7\\2406200640\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a4\\8\\2406140959\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a4\\9\\2406141238\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\mmse\\a5\\0\\2406200603\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a5\\1\\2406140551\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a5\\2\\2406140116\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a5\\3\\2406140054\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a5\\4\\2406141202\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a5\\5\\2406200704\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a5\\6\\2406140707\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a5\\7\\2406140059\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a5\\8\\2406140536\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a5\\9\\2406140840\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\mmse\\a6\\0\\2406141019\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a6\\1\\2406200959\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a6\\2\\2406140751\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a6\\3\\2406141256\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a6\\4\\2406141437\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a6\\5\\2406140301\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a6\\6\\2406201547\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a6\\7\\2406141003\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a6\\8\\2406141145\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\mmse\\a6\\9\\2406140039\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\mmse\\mt\\0\\2406241233\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\mt\\1\\2406240655\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\mt\\2\\2406241456\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\mt\\3\\2406241149\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\mt\\4\\2406240808\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\mt\\5\\2406240610\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\mt\\6\\2406241620\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\mt\\7\\2406241529\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\mt\\8\\2406240736\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\mmse\\mt\\9\\2406241610\model_best_rmse.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\moca\sss\\0\\2406080225\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\sss\\1\\2406101913\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\sss\\2\\2406091531\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\sss\\3\\2406100401\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\sss\\4\\2406120338\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\sss\\5\\2406100932\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\sss\\6\\2406111541\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\sss\\7\\2406072221\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\sss\\8\\2406090223\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\sss\\9\\2406051013\model_best_srocc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\moca\\a1\\0\\2406122216\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a1\\1\\2406131331\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a1\\2\\2406122219\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a1\\3\\2406130950\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a1\\4\\2406130646\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a1\\5\\2406131225\model_best_srocc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a1\\6\\2406130940\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a1\\7\\2406122321\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a1\\8\\2406130334\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a1\\9\\2406121705\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\moca\\a2\\0\\2406141022\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a2\\1\\2406141058\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a2\\2\\2406251446\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a2\\3\\2406251807\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a2\\4\\2406251534\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a2\\5\\2406141031\model_best_srocc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a2\\6\\2406140900\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a2\\7\\2406141316\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a2\\8\\2406141246\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a2\\9\\2406141214\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\moca\\a3\\0\\2406141049\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a3\\1\\2406140358\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a3\\2\\2406140431\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a3\\3\\2406140159\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\a3\\4\\2406141300\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a3\\5\\2406140659\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\a3\\6\\2406251429\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a3\\7\\2406140032\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a3\\8\\2406141205\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a3\\9\\2406140006\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\moca\\a4\\0\\2406141321\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\a4\\1\\2406141023\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a4\\2\\2406211634\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a4\\3\\2406210053\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a4\\4\\2406200443\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a4\\5\\2406141114\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a4\\6\\2406140633\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a4\\7\\2406141501\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a4\\8\\2406141248\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a4\\9\\2406210240\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\moca\\a5\\0\\2406140539\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a5\\1\\2406140343\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\a5\\2\\2406140214\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a5\\3\\2406141349\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a5\\4\\2406140823\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a5\\5\\2406140217\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a5\\6\\2406141242\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a5\\7\\2406210005\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a5\\8\\2406140851\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a5\\9\\2406140659\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\moca\\a6\\0\\2406140602\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a6\\1\\2406140714\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\a6\\2\\2406140449\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a6\\3\\2406140514\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a6\\4\\2406201630\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a6\\5\\2406140608\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a6\\6\\2406200532\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a6\\7\\2406140027\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a6\\8\\2406140114\model_best_plcc.pth.tar",
#                                            "logs\\ad_10f4\moca\\a6\\9\\2406141315\model_best_plcc.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\ad_10f4\moca\\mt\\0\\2406240418\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\mt\\1\\2406240626\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\mt\\2\\2406241052\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\mt\\3\\2406240504\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\mt\\4\\2406241359\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\mt\\5\\2406240628\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\mt\\6\\2406241531\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\mt\\7\\2406281818\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\mt\\8\\2406241029\model_best_rmse.pth.tar",
#                                            "logs\\ad_10f4\moca\\mt\\9\\2406241123\model_best_rmse.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
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
    ...

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
    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.evaluate:
        plcc_vals = []
        srocc_vals = []
        icls_vals = []
        outpus_vals = []
        labels_vals = []
        # weight_vals = []

        for i, ckp_path in enumerate(args.test_ckp):
            ETdata_test = folds[i]
            test_subset = Subset(full_dataset, ETdata_test)
            testloader = DataLoader(test_subset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            checkpoint = torch.load(ckp_path)
            model.load_state_dict(checkpoint['state_dict'])
            print('\nEvaluation only for checkpoint:', ckp_path)
            plcc_val, p_plcc, srocc_val, p_srocc, rmse_val, show_output, show_label, icls_all = test(testloader, model, use_cuda)
            print(show_output)
            plcc_vals.append(plcc_val)
            srocc_vals.append(srocc_val)
            icls_vals.append(icls_all)
            outpus_vals.append(show_output)
            labels_vals.append(show_label)
            # weight_vals.append(weight_mean)
        icls_vals = np.concatenate(icls_vals, axis=1).squeeze()
        outpus_vals = np.concatenate(outpus_vals, axis=1).squeeze()
        labels_vals = np.concatenate(labels_vals, axis=1).squeeze()
        mean_plcc = np.mean(plcc_vals)
        std_plcc = np.std(plcc_vals)
        mean_srocc = np.mean(srocc_vals)
        std_srocc = np.std(srocc_vals)
        # weight_cat = np.concatenate(weight_vals, axis=0)
        # print(weight_cat.shape)
        # weight_mean = np.mean(weight_cat, axis=0)
        # weight_split = np.split(weight_cat, 4, axis=1)
        # print(f'Weight Mean: {weight_mean}')

        
        print(f'Mean PLCC: {mean_plcc}, Std PLCC: {std_plcc}')
        print(f'Mean SROCC: {mean_srocc}, Std SROCC: {std_srocc}')
        # Convert outputs and targets to numpy arrays for plotting
        # show_output = np.concatenate(show_output)
        # show_label = np.concatenate(show_label)
        save_root = os.path.join('experiments/adreg', args.arch)
        if save_root:
            if not os.path.exists(save_root):
                os.makedirs(save_root)

        if args.questionnaire == 'mmse':
            try:
                all_preds_df = pd.read_csv('experiments/adreg/adreg_mmse.csv')
            except FileNotFoundError:
                all_preds_df = pd.DataFrame()
            
            all_preds_df[args.arch] = outpus_vals
            all_preds_df.to_csv('experiments/adreg/adreg_mmse.csv', index=False)
            
            plot_corr(labels_vals, outpus_vals, icls_vals, title='Correlation between MMSE and Model Prediction', 
                    x_label='MMSE Score', y_label='Model Prediction Score', save_path=os.path.join(save_root, 'mmse_reg.png'),
                    r_value=mean_plcc)
            
        elif args.questionnaire == 'moca':
            try:
                all_preds_df = pd.read_csv('experiments/adreg/adreg_moca.csv')
            except FileNotFoundError:
                all_preds_df = pd.DataFrame()
            
            all_preds_df[args.arch] = outpus_vals
            all_preds_df.to_csv('experiments/adreg/adreg_moca.csv', index=False)

            
            plot_corr(labels_vals, outpus_vals, icls_vals, title='Correlation between MOCA and Model Prediction', 
                      x_label='MOCA Score', y_label='Model Prediction Score', save_path=os.path.join(save_root, 'moca_reg.png'),
                      r_value=mean_plcc)
        
        # save evaluation results
        try:
            all_df = pd.read_csv('experiments/ablation/adreg.csv')
        except FileNotFoundError:
            all_df = pd.DataFrame()
        
        all_df[args.arch + '_' + args.questionnaire] = plcc_vals
        all_df.to_csv('experiments/ablation/adreg.csv', index=False)
       

def test(testloader, model, use_cuda):
    model.eval()
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    outputs_all = []
    targets_all = []
    icls_all = []
    # weights_all = []
    # show_output = []
    # show_label = []
    start_time = time.time()

    bar = Bar('Valid', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (gazemaps, taskmaps, age_edu, targets, icls) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - start_time)

            if use_cuda:
                gazemaps, taskmaps, age_edu, targets = [x.cuda() for x in gazemaps], [x.cuda() for x in taskmaps], age_edu.cuda(), targets.cuda()
                icls = icls.cuda()
            # compute output
            outputs = model(gazemaps, taskmaps, age_edu)
            # print(weight.shape)\
            # show_output.append(outputs.cpu().numpy())
            # show_label.append(targets.cpu().numpy())
            outputs_all.append(outputs.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())
            icls_all.append(icls.detach().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        )
            bar.next()
        bar.finish()
    plcc_val, p_plcc = calculate_plcc(np.concatenate(outputs_all), np.concatenate(targets_all))
    srocc_val, p_srocc = calculate_srocc(np.concatenate(outputs_all), np.concatenate(targets_all))
    rmse_val = calculate_rmse(np.concatenate(outputs_all), np.concatenate(targets_all))
    # weight_mean = np.concatenate(weights_all, axis=0).squeeze(-1)
    print('Total Time: {:.3f}s | PLCC: {:.4f} | P_PLCC: {:.4f} | SROCC: {:.4f} | P_SROCC: {:.4f} | RMSE: {:.4f}'.format(time.time() - start_time, plcc_val, p_plcc, srocc_val, p_srocc, rmse_val))
    return plcc_val, p_plcc, srocc_val, p_srocc, rmse_val, outputs_all, targets_all, icls_all

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


if __name__ == '__main__':
    main()