import os
import torch
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    #     print(top_pred.shape)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def class_accuracy(test_loader, model, args):
    """ Computes the accuracy for each class"""

    classes = args.class_names
    num_class = len(args.class_names)
    with torch.no_grad():
        n_class_correct = [0 for _ in range(num_class)]
        n_class_samples = [0 for _ in range(num_class)]
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            output = model(images)

            if args.logit_adj_post:
                output = output - args.logit_adjustments

            _, predicted = torch.max(output, 1)

            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        results = {}
        avg_acc = 0
        for i in range(num_class):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            avg_acc += acc
            results["class/" + classes[i]] = acc
        results["AA"] = avg_acc / num_class
        return results
    
def make_dir(log_dir):
    """ Makes a directory """

    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

