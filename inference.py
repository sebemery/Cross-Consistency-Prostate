import argparse
import numpy as np
import sys
import json
import models
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import base
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import PIL
import nibabel as nib
from utils.metrics import eval_metrics, AverageMeter
import cv2 as cv


def main():
    # get the argument from parser
    args = parse_arguments()

    # CONFIG -> assert if config is here
    assert args.config
    config = json.load(open(args.config))

    # DATA
    testdataset = base.testDataset(args.site)
    loader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)
    num_classes = testdataset.num_classes

    # MODEL
    config['model']['supervised'] = True; config['model']['semi'] = False
    encoder = models.model.Encoder(True)
    model = models.model.CCT(encoder, num_classes=num_classes, conf=config['model'], testing=True)
    map_location = args.map
    checkpoint = torch.load(args.model, map_location)

    if map_location == 'cpu':
        for key in list(checkpoint['state_dict'].keys()):
            if 'module.' in key:
                checkpoint['state_dict'][key.replace('module.', '')] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]

    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.float()
    model.eval()
    if args.map == 'gpu':
        model.cuda()

    check_directory(args.site, args.experiment)

    # LOOP OVER THE DATA
    tbar = tqdm(loader, ncols=100)

    total_loss_val = AverageMeter()
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0
    total_dice = 0
    count = 0

    for index, data in enumerate(tbar):
        image, label, image_id = data
        if args.map == 'gpu':
            image = image.cuda()

        # PREDICT
        with torch.no_grad():
            output = model(image)
            correct, labeled, inter, union, dice = eval_metrics(output, label, num_classes, args.map)
            total_inter, total_union = total_inter + inter, total_union + union
            total_correct, total_label = total_correct + correct, total_label + labeled
            total_dice = ((count * total_dice) + (dice * output.size(0))) / (count + output.size(0))
            count += output.size(0)
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            mdice = total_dice.mean()
            seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                           "Mean_dice": np.round(mdice, 3),
                           "Class_IoU": dict(zip(range(num_classes), np.round(IoU, 3))),
                           "Class_dice": dict(zip(range(num_classes), np.round(total_dice, 3)))}
            tbar.set_description('EVAL | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} Mean Dice {:.2f} |'.format(
                total_loss_val.average, pixAcc, mIoU, mdice))
            output = torch.argmax(output, dim=1)
        prediction = output.numpy()
        label = label.numpy()
        predictions = batch_scale(prediction)
        labels = batch_scale(label)

        if args.overlay:
            prediction_contours = batch_contour(predictions)
            label_contours = batch_contour(labels)

        # SAVE RESULTS
        for i in range(predictions.shape[0]):
            prediction_im = PIL.Image.fromarray(predictions[i])
            prediction_im.save(f'outputs/{args.site}/{args.experiment}/{image_id[i]}_prediction.png')
            label_im = PIL.Image.fromarray(labels[i])
            label_im.save(f'outputs/{args.site}/{args.experiment}/{image_id[i]}_label.png')

        if args.overlay:
            image = image.numpy()
            image = np.squeeze(image, axis=1)
            image = batch_scale(image)
            palette = contour_palette(testdataset.site)
            for i in range(image.shape[0]):
                image_gt = cv.cvtColor(image[i].copy(), cv.COLOR_GRAY2RGB)
                image_pred = cv.cvtColor(image[i].copy(), cv.COLOR_GRAY2RGB)
                cv.drawContours(image_gt, label_contours[i], -1, (palette[0], palette[1], palette[2]), 1)
                cv.drawContours(image_pred, prediction_contours[i], -1, (palette[0], palette[1], palette[2]), 1)
                cv.imwrite(f'outputs/{args.site}/{args.experiment}/{image_id[i]}_label_overlay.png', image_gt)
                cv.imwrite(f'outputs/{args.site}/{args.experiment}/{image_id[i]}_prediction_overlay.png', image_pred)

    with open(f'outputs/{args.site}/{args.experiment}/test.txt', 'w') as f:
        for k, v in list(seg_metrics.items()):
            f.write("%s\n" % (k + ':' + f'{v}'))


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='configs/config.json', type=str,
                        help='Path to the config file')
    parser.add_argument('--model', default=None, type=str,
                        help='Path to the trained .pth model')
    parser.add_argument('--site', default="BIDMC", type=str,
                        help='site to test')
    parser.add_argument('--map', default="cpu", type=str,
                        help='map location')
    parser.add_argument('--experiment', default=None, type=str,
                        help='experiment name')
    parser.add_argument('--overlay', default=False, type=bool,
                        help='return original image with overlay of the ground truth and predicted segmentation')
    args = parser.parse_args()
    return args


def check_directory(site, experiment):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists(f'outputs/{site}'):
        os.makedirs(f'outputs/{site}')
    if not os.path.exists(f'outputs/{site}/{experiment}'):
        os.makedirs(f'outputs/{site}/{experiment}')


def batch_scale(image):
    for i, img in enumerate(image):
        a = np.amax(img) - np.amin(img)

        if a == 0:
            if np.amax(img) == 2:
                img = img / 2
                image[i, :, :] = 255 * img
            else:
                image[i, :, :] = 255 * img
        else:
            img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
            image[i, :, :] = 255 * img
    return np.uint8(image)


def batch_contour(image):
    contours = []
    for i, img in enumerate(image):
        prediction_contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours.append(prediction_contours)
    return contours


def denormalize(image, mean, std):
    image = (image * std) + mean
    return image


def contour_palette(site):
    palette = {'ISBI': [0, 0, 255], 'ISBI_15': [0, 255, 0], 'I2CVB': [255, 0, 0],
               'BIDMC': [0, 125, 255], 'HK': [0, 255, 255], 'UCL': [255, 0, 125]}
    return palette[site]


if __name__ == '__main__':
    main()

