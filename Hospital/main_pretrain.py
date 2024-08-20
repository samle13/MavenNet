import os
import torch
from tqdm import tqdm

import torch.backends.cudnn as cudnn
from torch import optim
from torch import nn

from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve  # 返回fpr、tpr、threshhold
from sklearn.metrics import roc_auc_score  # 返回ROC曲线下的面积
from sklearn.metrics import auc  # 返回ROC曲线下的面积
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score
# from sklearn.metrics import plot_roc_curve  # 用于绘制ROC曲线

import numpy as np
from scipy import io
from numpy import *
from matplotlib import pyplot as pyplot
import torchvision.models as models


from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import logging
import random
import time
import utils

from config import get_arguments
from model import CNN2DModel
from get_data import get_loaders

import torch.optim as optim

parser = get_arguments()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device


args.log_path = os.path.join(args.log_base, args.model, f'S_{args.scales}',  f'seed_{args.seed}')
args.save_model = False

# 创建SummaryWriter对象，指定日志存储路径

writer = SummaryWriter(args.log_path)

logging.basicConfig(filename=f'{args.log_path}/1.txt', level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

def modify_first_layer(model, input_channels):
    # 获取模型的第一个卷积层
    if hasattr(model, 'conv1'):
        conv1 = model.conv1
        model.conv1 = nn.Conv2d(input_channels, conv1.out_channels, kernel_size=conv1.kernel_size, 
                                stride=conv1.stride, padding=conv1.padding, bias=conv1.bias is not None)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    elif hasattr(model, 'features') and isinstance(model.features[0], nn.Conv2d):
        conv1 = model.features[0]
        model.features[0] = nn.Conv2d(input_channels, conv1.out_channels, kernel_size=conv1.kernel_size, 
                                      stride=conv1.stride, padding=conv1.padding, bias=conv1.bias is not None)
        nn.init.kaiming_normal_(model.features[0].weight, mode='fan_out', nonlinearity='relu')
    elif hasattr(model, 'features') and isinstance(model.features[0], nn.Sequential):
        conv1 = model.features[0][0]
        model.features[0][0] = nn.Conv2d(input_channels, conv1.out_channels, kernel_size=conv1.kernel_size, 
                                         stride=conv1.stride, padding=conv1.padding, bias=conv1.bias is not None)
        nn.init.kaiming_normal_(model.features[0][0].weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(model, models.GoogLeNet):
        conv1 = model.conv1.conv
        model.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, conv1.out_channels, kernel_size=conv1.kernel_size, 
                      stride=conv1.stride, padding=conv1.padding, bias=conv1.bias is not None),
            nn.BatchNorm2d(conv1.out_channels),
            nn.ReLU(inplace=True)
        )
        nn.init.kaiming_normal_(model.conv1[0].weight, mode='fan_out', nonlinearity='relu')
    else:
        raise ValueError("Unsupported model architecture for modifying the first layer.")
    return model

def get_pretrained_model(model_name, num_classes, input_channels):
    if model_name == 'resnet':
        model = models.resnet18(pretrained=True)
        model = modify_first_layer(model, input_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'googlenet':
        print(input_channels)
        model = models.googlenet(pretrained=True)
        # Modify the first convolutional layer to match input_channels
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final layer to match num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
        model = modify_first_layer(model, input_channels)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Modify the first convolutional layer to match input_channels
        model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        # Modify the final layer to match num_classes
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Unsupported model name. Choose from 'resnet', 'googlenet', 'mobilenet'")
    
    return model

def set_seed(seed):
    # fix seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(args.seed)

if args.set == 'no_overlap':
    args.data_dir = "./data/no_overlap"
elif args.set == 'overlap':
    args.data_dir =f'./data/S_{args.scales}/overlap'


input_channels = 19

def main():
    train_loader, test_loader = get_loaders(args)
    model = get_pretrained_model(args.model, args.num_class, input_channels)
    model = model.to(device)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    loop = tqdm(range(0, args.epochs), total=args.epochs, leave=False)

    test_loss, test_acc = 0, 0
    for epoch in loop:

        start_time = time.monotonic()
        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("train/loss", train_loss, epoch)

        test_loss, test_acc, f1_weighted, sensitivity, specificity, roc_auc, kappa, precision, recall = evaluate(model, test_loader, criterion)


        writer.add_scalar("test/acc", test_acc, epoch)
        writer.add_scalar("test/loss", test_loss, epoch)


        logging.info(f'Epoch{epoch + 1}/{args.epochs}:')
        logging.info(f'\tTrain Loss:{train_loss:.4f}| Train Acc:{train_acc * 100:.2f}% ')
        logging.info(f'\tTest Loss:{test_loss:.4f}| Test Acc:{test_acc * 100:.2f}% ')
        logging.info(f'\tWeigthed F1:{f1_weighted:.4f}| ROC score:{roc_auc:.4f} | ')
        logging.info(f'\tKappa: {kappa:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}')
        logging.info(f'\t Specificity: {specificity:.4f} | Sensitivity:{sensitivity:.4f}')
        if args.save_model == True:

            model_save_path = os.path.join(args.model_save_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': test_loss
                    }, model_save_path
                )
        end_time = time.monotonic()

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%')

        loop.set_description(f"Epoch [{epoch}/{args.epochs}")
        loop.set_postfix(train_loss=f"{train_loss:.2f}", test_loss=f"{test_loss:.2f}",
                         train_acc=f"{train_acc:.2f}",
                         test_acc=f"{test_acc:.2f}")
        scheduler.step()

def train(train_loader, model, criterion, optimizer):
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        start_time_sigle = time.monotonic()
        # print(x.shape)
        x = x.permute(0,4,2,3,1).squeeze(1)
        # x = x.permute(0, 2, 1).unsqueeze(1)
        '''
        每条数据的特征维数从[16,127,19,500,5]reshape为[16,19,500,127,5]
        '''        
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)

        # y_pred = y_pred + args.logit_adjustment
        #         print(y_pred.shape, y.shape)
        loss = criterion(y_pred, y)
        acc = utils.calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        losses.update(loss.item(), x.size(0))
        accuracies.update(acc, y.size(0))

        end_time_sigle = time.monotonic()

        sigle_secs = end_time_sigle - start_time_sigle
        print(f'Batch [{batch_idx}/{len(train_loader)}] | '
                  f'Loss: {loss.item():.3f} | '
                  f'Acc: {acc.item() * 100:.2f}% | 'f'Batch Size: {args.batch_size}') 

    return losses.avg, accuracies.avg

def evaluate(model, iterator, criterion):
    """ Run evaluation """

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter() 

    all_predictions = []
    all_labels = []
    all_probs = []


    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(iterator):
            # x = x.permute(0, 2, 3, 1, 4)
            # x = x.permute(0, 2, 1).unsqueeze(1)
            x = x.permute(0,4,2,3,1).squeeze(1)
            '''

            每条数据的特征维数从[16,127,19,500,5]reshape为[16,19,500,127,5]
            '''
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            acc = utils.calculate_accuracy(y_pred, y)

            losses.update(loss.item(), x.size(0))
            accuracies.update(acc, x.size(0))

            batch_size = x.size(0)

            # 记录预测结果和真实标签
            all_predictions.extend(torch.argmax(y_pred, axis=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(torch.softmax(y_pred, dim=1).cpu().numpy()) 
            # 记录训练损失


            print(f'Validation Batch [{batch_idx}/{len(iterator)}] | '
                    f'Loss: {loss.item():.3f} | '
                    f'Acc: {acc.item() * 100:.2f}% | '
                    f'Batch Size: {args.batch_size}')
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)

        print('Confusion Matrix:')
        print(cm)
        # 计算加权F1得分
        f1_weighted = f1_score(np.argmax(cm, axis=1), np.argmax(cm, axis=0), average='weighted')
        print(f'加权F1得分:{f1_weighted:.4f}')
        sensitivity_per_class = recall_score(all_labels, all_predictions, average=None)
        macro_averaged_sensitivity = np.mean(sensitivity_per_class)

        
        print("Macro-Averaged Sensitivity:", macro_averaged_sensitivity)
        
        # 计算特异性
        specificity_per_class = []
        for i in range(len(cm)):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm, i, axis=0)[:, i])
            specificity = tn / (tn + fp)
            specificity_per_class.append(specificity)
        macro_averaged_specificity = np.mean(specificity_per_class)
        print("Macro-Averaged Specificity:", macro_averaged_specificity)


        # 计算ROC AUC得分
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        print(f'ROC AUC Score: {roc_auc:.4f}')
        
        # 计算Kappa系数
        kappa = cohen_kappa_score(all_labels, all_predictions)
        print(f'Kappa Score: {kappa:.4f}')
        
        # 计算Precision
        precision = precision_score(all_labels, all_predictions, average='weighted')
        print(f'Precision: {precision:.4f}')
        
        # 计算Recall
        recall = recall_score(all_labels, all_predictions, average='weighted')
        print(f'Recall: {recall:.4f}')


        return losses.avg, accuracies.avg, f1_weighted, macro_averaged_sensitivity, macro_averaged_specificity, roc_auc, kappa, precision, recall


if __name__ == '__main__':
    main()
