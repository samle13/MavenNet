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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import numpy as np
from scipy import io
from numpy import *
from matplotlib import pyplot as pyplot

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import logging
import random
import time
import utils

from sklearn.metrics import recall_score

from config import get_arguments
from model import CNN2DModel
from get_data import get_loaders

parser = get_arguments()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

args.log_path = os.path.join(args.log_base,args.set ,f'seed_{args.seed}')


# 创建SummaryWriter对象，指定日志存储路径

writer = SummaryWriter(args.log_path)

if args.set == 'ABvsCDvsE':
    args.data_dir = f"./data/{args.overlap}/S_{args.scales}/{args.set}"
elif args.set =='ABvsCDE':
    args.data_dir = f"./data/{args.overlap}/S_{args.scales}/{args.set}"


logging.basicConfig(filename=f'{args.log_path}/1.txt', level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

# 固定随机种子

def set_seed(seed):
    # fix seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(args.seed)

# print(args)
def main():
    print(args)
    train_loader, test_loader = get_loaders(args)
    model = CNN2DModel(3)
    model = model.to(device)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
        if args.save_model:
            args.model_save_dir = f'models/{args.set}/{args.scales}'
            model_save_path = os.path.join(args.model_save_dir, f'model_epoch_{epoch + 1}.pt')
            if not os.path.exists(args.model_save_dir):
                os.makedirs(args.model_save_dir)

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

def test():
    model_path = 'models/model_epoch_50.pt'  # 替换为实际的模型文件路径

    train_loader, test_loader = get_loaders(args)
    model = CNN2DModel(args.num_class).to(device)
    # 加载已保存的模型状态字典
    checkpoint = torch.load(model_path)

    # 从检查点中恢复模型状态
    model.load_state_dict(checkpoint['model_state_dict'])

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)

    test_loss, test_acc = 0, 0



    test_loss, test_acc = evaluate(model, test_loader, criterion)


    print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%')

def plot():
    model_path = 'models/ABvsCDvsE/64/model_epoch_48.pt'  # 替换为实际的模型文件路径

    train_loader, test_loader = get_loaders(args)
    print(args)
    model = CNN2DModel(3).to(device)
    # 加载已保存的模型状态字典
    checkpoint = torch.load(model_path)

    # 从检查点中恢复模型状态
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    # 保存所有特征和标签
    features = []
    labels = []

    # 不需要计算梯度
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            # print(data.shape)
            data = data.permute(0, 1, 3, 2)
            output = model(data, use_feature=True)  # 获取卷积层特征
            features.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())

    # 将列表转化为numpy数组
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='jet', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE of Test Dataset Features')
    plt.show()

def train(train_loader, model, criterion, optimizer):
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    
    model.train()
    


    for batch_idx, (x, y) in enumerate(train_loader):
        start_time_sigle = time.monotonic()
        
        x = x.permute(0, 1, 3, 2)
    
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)

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
            x = x.permute(0, 1, 3, 2)

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
            all_probs.extend(torch.softmax(y_pred, dim=1).cpu().numpy())  # 获取每个类别的概率
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
        f1_weighted = f1_score(np.argmax(cm, axis=0), np.argmax(cm, axis=1), average='weighted')
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

        # all_probs_positive_class = [prob[1] for prob in all_probs]
        # roc_auc = roc_auc_score(all_labels, all_probs_positive_class)
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
    # test()
    # main()
    plot()
