import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


def batch_pix_accuracy(output, target):
    predict = torch.argmax(output, dim=1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def batch_intersection_union(output, target, num_class):
    predict = torch.argmax(output, dim=1)
    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()


def eval_metrics(output, target, num_classes, device):
    target = target.clone()
    correct, labeled = batch_pix_accuracy(output.data, target)
    inter, union = batch_intersection_union(output.data, target, num_classes)
    dice = dice_score(output.data, target, device)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5), np.round(dice, 5)]


# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(output, target):
    output = np.asarray(output)
    target = np.asarray(target)
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((output == target) * (target > 0))
    return pixel_correct, pixel_labeled


def inter_over_union(output, target, num_class):
    output = np.asarray(output) + 1
    target = np.asarray(target) + 1
    output = output * (target > 0)

    intersection = output * (output == target)
    area_inter, _ = np.histogram(intersection, bins=num_class, range=(1, num_class))
    area_pred, _ = np.histogram(output, bins=num_class, range=(1, num_class))
    area_lab, _ = np.histogram(target, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


def dice_score(output, target, device):
    """
    output is a torch variable of size NxCxHxW
    target tensor Nx1XHxW which is transformed into one-hot target NxCxHxW
    """

    target = target.unsqueeze(1)
    preds = torch.argmax(output, dim=1)
    preds = torch.unsqueeze(preds, dim=1)
    # Convert target into one-hot vector
    target_onehot = torch.FloatTensor(output.shape)
    preds_onehot = torch.FloatTensor(output.shape)
    if device == 'gpu':
        target_onehot = target_onehot.cuda(non_blocking=True)
        preds_onehot = preds_onehot.cuda(non_blocking=True)
    target_onehot.zero_()
    preds_onehot.zero_()
    ones = torch.ones(target.shape)
    ones_preds = torch.ones(target.shape)
    if device == 'gpu':
        ones = ones.cuda(non_blocking=True)
        ones_preds = ones_preds.cuda(non_blocking=True)
    target_onehot.scatter_(1, target, ones)
    preds_onehot.scatter_(1, preds, ones_preds)

    # compute intersection
    num = preds_onehot * target_onehot  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)  # b,c,h
    num = torch.sum(num, dim=2)

    # compute element probs
    den1 = preds_onehot
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)

    # compute element target
    den2 = target_onehot
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)

    dice = 2 * (num / (np.spacing(1)+den1 + den2))

    return dice.cpu().numpy().mean(axis=0)
