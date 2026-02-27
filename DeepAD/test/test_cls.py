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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from utils.visualize import boxplot, boxplot2, plot_roc, violinplot2, plot_roc2

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='PyTorch AD Training')

# 数据集
parser.add_argument('-d', '--dataset', default='/media/mprl2/Hard Disk/zwl/ADR2/gazedata240507', type=str)
parser.add_argument('--datacsv', default='adnc_10f4', type=str, help='dataset .csv file')
parser.add_argument('--hdf5_file', default='data.hdf5', type=str, help='dataset .csv file')
parser.add_argument('--questionnaire', default='cls', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='数据加载工作进程数 (默认: 4)')

# 优化选项
parser.add_argument('--test-batch', default=20, type=int, metavar='N',
                    help='测试批次大小')

# 模型架构
parser.add_argument('--arch', '-a', metavar='ARCH', default='mt',
                    choices=model_names,
                    help='模型架构: ' +
                        ' | '.join(model_names) +
                        ' (默认: scls,ac1,ac2,ac3,ac4,ac5,ac6,ac7,ac8,ac9)')
parser.add_argument('--aucname', default='Multitask', type=str, help='auc name(Saccade task, Visual search task, Visual attention task, Multitask)')
# 杂项选项
parser.add_argument('--manualSeed', type=int, help='手动随机种子')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=bool, default=True,
                    help='仅在验证集上评估模型')
# parser.add_argument('--test_ckp', default=["logs\\adnc_10f4\cls\scls\\0\\2406221755\model_best_acc.pth.tar",
#                                            "logs\\adnc_10f4\cls\scls\\1\\2406161629\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\scls\\2\\2406170838\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\scls\\3\\2406190153\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\scls\\4\\2406221742\model_best_acc.pth.tar",
#                                            "logs\\adnc_10f4\cls\scls\\5\\2406161754\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\scls\\6\\2406212046\model_best_acc.pth.tar",
#                                            "logs\\adnc_10f4\cls\scls\\7\\2406062037\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\scls\\8\\2406161930\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\scls\\9\\2406161918\model_best.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\adnc_10f4\cls\\ac1\\0\\2406160757\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac1\\1\\2406160331\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac1\\2\\2406160721\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac1\\3\\2406160255\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac1\\4\\2406160414\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac1\\5\\2406160610\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac1\\6\\2406160349\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac1\\7\\2406160311\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac1\\8\\2406160319\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac1\\9\\2406160324\model_best.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\adnc_10f4\cls\\ac2\\0\\2406161300\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac2\\1\\2406162238\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac2\\2\\2406161430\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac2\\3\\2406161601\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac2\\4\\2406161336\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac2\\5\\2406161338\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac2\\6\\2406162355\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac2\\7\\2406161241\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac2\\8\\2406161228\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac2\\9\\2406161934\model_best.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\adnc_10f4\cls\\ac3\\0\\2406161706\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac3\\1\\2406161213\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac3\\2\\2406161507\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac3\\3\\2406161244\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac3\\4\\2406162023\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac3\\5\\2406161126\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac3\\6\\2406161319\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac3\\7\\2406161236\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac3\\8\\2406162100\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac3\\9\\2406161224\model_best.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\adnc_10f4\cls\\ac4\\0\\2406181523\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac4\\1\\2406170305\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac4\\2\\2406170511\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac4\\3\\2406170312\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac4\\4\\2406181447\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac4\\5\\2406171003\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac4\\6\\2406170723\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac4\\7\\2406170927\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac4\\8\\2406171032\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac4\\9\\2406170343\model_best.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\adnc_10f4\cls\\ac5\\0\\2406171315\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac5\\1\\2406171412\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac5\\2\\2406171353\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac5\\3\\2406181659\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac5\\4\\2406171541\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac5\\5\\2406171115\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac5\\6\\2406171406\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac5\\7\\2406171452\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac5\\8\\2406171426\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac5\\9\\2406171455\model_best.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\adnc_10f4\cls\\ac6\\0\\2406171139\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac6\\1\\2406171228\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac6\\2\\2406171936\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac6\\3\\2406180042\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac6\\4\\2406171405\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac6\\5\\2406171525\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac6\\6\\2406171723\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac6\\7\\2406171535\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac6\\8\\2406171118\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac6\\9\\2406171207\model_best.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\adnc_10f4\cls\\ac7\\0\\2406181751\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac7\\1\\2406181224\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac7\\2\\2406181317\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac7\\3\\2406181120\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac7\\4\\2406181435\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac7\\5\\2406181323\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac7\\6\\2406181249\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac7\\7\\2406181108\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac7\\8\\2406181203\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac7\\9\\2406181132\model_best.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\adnc_10f4\cls\\ac8\\0\\2406181640\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac8\\1\\2406181059\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac8\\2\\2406181159\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac8\\3\\2406181102\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac8\\4\\2406181251\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac8\\5\\2406181203\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac8\\6\\2406181538\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac8\\7\\2406181311\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac8\\8\\2406181130\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac8\\9\\2406181113\model_best.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
# parser.add_argument('--test_ckp', default=["logs\\adnc_10f4\cls\\ac9\\0\\2406181353\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac9\\1\\2406181319\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac9\\2\\2406181248\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac9\\3\\2406181535\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac9\\4\\2406181724\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac9\\5\\2406181217\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac9\\6\\2406181204\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac9\\7\\2406181109\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac9\\8\\2406181131\model_best.pth.tar",
#                                            "logs\\adnc_10f4\cls\\ac9\\9\\2406181257\model_best.pth.tar"], type=str, nargs='+', metavar='PATH',
#                     help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
parser.add_argument('--test_ckp', default=["logs\\adnc_10f4\cls\\mt\\0\\2406281913\model_best_acc.pth.tar",
                                           "logs\\adnc_10f4\cls\\mt\\1\\2406281733\model_best_acc.pth.tar",
                                           "logs\\adnc_10f4\cls\\mt\\2\\2406281626\model_best_acc.pth.tar",
                                           "logs\\adnc_10f4\cls\\mt\\3\\2406281636\model_best_acc.pth.tar",
                                           "logs\\adnc_10f4\cls\\mt\\4\\2406281941\model_best_acc.pth.tar",
                                           "logs\\adnc_10f4\cls\\mt\\5\\2406281917\model_best_acc.pth.tar",
                                           "logs\\adnc_10f4\cls\\mt\\6\\2406281630\model_best_acc.pth.tar",
                                           "logs\\adnc_10f4\cls\\mt\\7\\2406281651\model_best_acc.pth.tar",
                                           "logs\\adnc_10f4\cls\\mt\\8\\2406281714\model_best_acc.pth.tar",
                                           "logs\\adnc_10f4\cls\\mt\\9\\2406281731\model_best_acc.pth.tar"], type=str, nargs='+', metavar='PATH',
                    help='测试检查点的路径 (默认: checkpoint/model_best.pth.tar)')
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
        all_preds = []
        all_targets_binary = []
        all_preds_binary = []
        accs, precisions, recalls, f1s, roc_aucs, sensitivities, specificities = [], [], [], [], [], [], []
        weight_vals = []

        for i, ckp_path in enumerate(args.test_ckp):
            ETdata_test = folds[i]
            test_subset = Subset(full_dataset, ETdata_test)
            testloader = DataLoader(test_subset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            checkpoint = torch.load(ckp_path)
            model.load_state_dict(checkpoint['state_dict'])
            print('\nEvaluation only for checkpoint:', ckp_path)
            acc, precision, recall, f1, roc_auc, sensitivity, specificity, targets_binary, preds_binary, preds = test(testloader, model, use_cuda)
            all_preds.extend(preds)
            all_targets_binary.extend(targets_binary)
            all_preds_binary.extend(preds_binary)
            accs.append(acc)
            # precisions.append(precision)
            # recalls.append(recall)
            # f1s.append(f1)
            roc_aucs.append(roc_auc)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            # weight_vals.append(weight_mean)

        # Calculate mean and std for each metric
        acc_mean, acc_std = np.mean(accs), np.std(accs)

        # precision_mean, precision_std = np.mean(precisions), np.std(precisions)
        # recall_mean, recall_std = np.mean(recalls), np.std(recalls)
        # f1_mean, f1_std = np.mean(f1s), np.std(f1s)
        roc_auc_mean, roc_auc_std = np.mean(roc_aucs), np.std(roc_aucs)
        sensitivity_mean, sensitivity_std = np.mean(sensitivities), np.std(sensitivities)
        specificity_mean, specificity_std = np.mean(specificities), np.std(specificities)
        z = 1.96
        n_per_fold = 10
        sensitivity_ci = (sensitivity_mean - z * (sensitivity_std / np.sqrt(n_per_fold)),sensitivity_mean + z * (sensitivity_std / np.sqrt(n_per_fold)))
        specificity_ci = (specificity_mean - z * (specificity_std / np.sqrt(n_per_fold)),specificity_mean + z * (specificity_std / np.sqrt(n_per_fold)))
        # weight_cat = np.concatenate(weight_vals, axis=0)
        # print(weight_cat.shape)
        # weight_mean = np.mean(weight_cat, axis=0)
        # weight_split = np.split(weight_cat, 4, axis=1)
        # print(f'Weight Mean: {weight_mean}')

        print(f"Accuracy: Mean = {acc_mean}, Std = {acc_std}")
        # print(f"Precision: Mean = {precision_mean}, Std = {precision_std}")
        # print(f"Recall: Mean = {recall_mean}, Std = {recall_std}")
        # print(f"F1 Score: Mean = {f1_mean}, Std = {f1_std}")
        print(f"ROC-AUC: Mean = {roc_auc_mean}, Std = {roc_auc_std}")
        print(f"Sensitivity: Mean = {sensitivity_mean}, Std = {sensitivity_std}", f"95% CI = ({sensitivity_ci[0]:.4f}, {sensitivity_ci[1]:.4f})")
        print(f"Specificity: Mean = {specificity_mean}, Std = {specificity_std}", f"95% CI = ({specificity_ci[0]:.4f}, {specificity_ci[1]:.4f})")
        
        save_root = os.path.join('experiments/adnccls', args.arch)
        if save_root:
            if not os.path.exists(save_root):
                os.makedirs(save_root)
        try:
            all_preds_df = pd.read_csv('experiments/adnccls/adnccls.csv')
        except FileNotFoundError:
            all_preds_df = pd.DataFrame()
        
        all_preds_df[args.arch] = all_preds
        all_preds_df.to_csv('experiments/adnccls/adnccls.csv', index=False)

        # ## save evaluation results
        # if save_root:
        #     if not os.path.exists(save_root):
        #         os.makedirs(save_root)
        # try:
        #     all_df = pd.read_csv('experiments/ablation/Indicator.csv')
        # except FileNotFoundError:
        #     all_df = pd.DataFrame()
        
        # all_df[args.arch + '_auc'] = roc_aucs
        # all_df.to_csv('experiments/ablation/Indicator.csv', index=False)
        # all_df[args.arch + '_sens'] = sensitivities
        # all_df.to_csv('experiments/ablation/Indicator.csv', index=False)
        # all_df[args.arch + '_spec'] = specificities
        # all_df.to_csv('experiments/ablation/Indicator.csv', index=False)
        ## save evaluation results

        boxplot2(all_preds, all_targets_binary, save_path=os.path.join(save_root, 'boxline.png'))
        violinplot2(all_preds, all_targets_binary, save_path=os.path.join(save_root, 'violinline.png'))
        plot_roc(all_targets_binary, all_preds, roc_auc_mean, args.aucname,  save_path=os.path.join(save_root, 'roc.png'))


def test(testloader, model, use_cuda):
    model.eval()
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    preds_all = []
    targets_all = []
    weights_all = []
    start_time = time.time()

    bar = Bar('Valid', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (gazemaps, taskmaps, age_edu, targets, _) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - start_time)

            if use_cuda:
                gazemaps, taskmaps, age_edu, targets = [x.cuda() for x in gazemaps], [x.cuda() for x in taskmaps], age_edu.cuda(), targets.cuda()

            # compute output
            outputs = model(gazemaps, taskmaps, age_edu)
            preds = torch.sigmoid(outputs)

            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())  
            # weights_all.append(weights.detach().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - start_time)

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        )
            bar.next()
        bar.finish()
    preds_binary = (np.concatenate(preds_all) > 0.5).astype(int)
    targets_binary = np.concatenate(targets_all).astype(int)
    acc = accuracy_score(targets_binary, preds_binary)
    precision = precision_score(targets_binary, preds_binary)
    recall = recall_score(targets_binary, preds_binary)
    f1 = f1_score(targets_binary, preds_binary)
    roc_auc = roc_auc_score(np.concatenate(targets_all), np.concatenate(preds_all))
    sensitivity, specificity = cal_sens_spec(targets_binary, preds_binary)

    # Calculate 95% confidence interval for sensitivity and specificity
    # sensitivity_se = np.sqrt((sensitivity * (1 - sensitivity)) / len(targets_binary))
    # specificity_se = np.sqrt((specificity * (1 - specificity)) / len(targets_binary))
    # sensitivity_conf_int = stats.norm.interval(0.95, loc=sensitivity, scale=sensitivity_se)
    # specificity_conf_int = stats.norm.interval(0.95, loc=specificity, scale=specificity_se)

    # print('Sensitivity 95% Confidence Interval: {:.4f} to {:.4f}'.format(sensitivity_conf_int[0], sensitivity_conf_int[1]))
    # print('Specificity 95% Confidence Interval: {:.4f} to {:.4f}'.format(specificity_conf_int[0], specificity_conf_int[1]))
    print('Total Time: {:.3f}s | ACC: {:.4f} | Sensitivity: {:.4f} | Specificity: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f} | ROC-AUC: {:.4f}'.format(time.time() - start_time, acc, sensitivity, specificity, precision, recall, f1, roc_auc))
    # Save ROC curve image

    return acc, precision, recall, f1, roc_auc, sensitivity, specificity, targets_binary, preds_binary, np.concatenate(preds_all)


def cal_sens_spec(targets, preds):
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    # Calculate sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

if __name__ == '__main__':
    main()
