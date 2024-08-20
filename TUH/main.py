import os
import torch
from tqdm import tqdm

import torch.backends.cudnn as cudnn
from torch import optim
from torch import nn

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

from config import get_arguments
from model import CNN2DModel
from get_data import get_loaders

parser = get_arguments()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

# # 创建SummaryWriter对象，指定日志存储路径
# log_dir = './1M32S'
# writer = SummaryWriter(log_dir)
#
# os.makedirs(args.model_save_dir, exist_ok= True)
# logging.basicConfig(filename=args.log_filename, level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

# 固定随机种子
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def main():
    train_loader, test_loader, patient_seizure_matrix = get_loaders(args)
    patient_seizure_matrix = torch.tensor(patient_seizure_matrix,dtype=torch.float32).to(device)
    model = CNN2DModel(args.num_class)
    model = model.to(device)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    loop = tqdm(range(0, args.epochs), total=args.epochs, leave=False)

    test_loss, test_acc = 0, 0
    for epoch in loop:
        start_time = time.monotonic()
        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, patient_seizure_matrix)
        # writer.add_scalar("train/acc", train_acc, epoch)
        # writer.add_scalar("train/loss", train_loss, epoch)

        test_loss, test_acc = evaluate(model, test_loader, criterion, patient_seizure_matrix)
        #
        # writer.add_scalar("test/acc", test_acc, epoch)
        # writer.add_scalar("test/loss", test_loss, epoch)

        #
        # logging.info(f'Epoch{epoch + 1}/{args.epochs}:')
        # logging.info(f'\tTrain Loss:{train_loss:.4f}| Train Acc:{train_acc * 100:.2f}% ')
        # logging.info(f'\tTest Loss:{test_loss:.4f}| Test Acc:{test_acc * 100:.2f}% ')
        # model_save_path = os.path.join(args.model_save_dir, f'model_epoch_{epoch + 1}.pt')
        # torch.save(
        #         {
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': test_loss
        #         }, model_save_path
        #     )
        end_time = time.monotonic()

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%')

        loop.set_description(f"Epoch [{epoch}/{args.epochs}")
        loop.set_postfix(train_loss=f"{train_loss:.2f}", test_loss=f"{test_loss:.2f}",
                         train_acc=f"{train_acc:.2f}",
                         test_acc=f"{test_acc:.2f}")


def train(train_loader, model, criterion, optimizer, patient_seizure_matrix):
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    model.train()
    for batch_idx, (x, z, y) in enumerate(train_loader):
        start_time_sigle = time.monotonic()


        x = x.squeeze(1).permute(0, 2, 3, 1)


        x = x.to(device)
        y = y.to(device)
        z = z.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss_seizure = torch.nn.functional.nll_loss(torch.mm(torch.log(torch.softmax(y_pred, dim=1)), patient_seizure_matrix),z)
        # y_pred = y_pred + args.logit_adjustment
        #         print(y_pred.shape, y.shape)
        loss_patient = criterion(y_pred, y)
        acc = utils.calculate_accuracy(y_pred, y)
        loss = 0.2*loss_seizure + 0.8*loss_patient
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


def evaluate(model, iterator, criterion, patient_seizure_matrix):
    """ Run evaluation """

    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    all_predictions = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(iterator):
            # x = x.permute(0, 2, 3, 1, 4)
            x = x.squeeze(1).permute(0, 2, 3, 1)
            '''

            每条数据的特征维数从[16,127,19,500,5]reshape为[16,19,500,127,5]
            '''
            x = x.to(device)
            y = y.to(device)

            z_pred = model(x)
            # print(z_pred.shape)
            y_pred = torch.mm(torch.log(torch.softmax(z_pred, dim=1)), patient_seizure_matrix)
            # print(y_pred.shape)
            loss = torch.nn.functional.nll_loss(y_pred, y)

            acc = utils.calculate_accuracy(y_pred, y)

            losses.update(loss.item(), x.size(0))
            accuracies.update(acc, x.size(0))

            batch_size = x.size(0)

            # 记录预测结果和真实标签
            all_predictions.extend(torch.argmax(y_pred, axis=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            # 记录训练损失

            print(f'Validation Batch [{batch_idx}/{len(iterator)}] | '
                  f'Loss: {loss.item():.3f} | '
                  f'Acc: {acc.item() * 100:.2f}% | '
                  f'Batch Size: {args.batch_size}')
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)

        print('Confusion Matrix:')
        print(cm)

        return losses.avg, accuracies.avg


if __name__ == '__main__':
    main()
