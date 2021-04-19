import os
import json
import argparse
import torch
import dataloaders
import models
import math
from utils import Logger
from trainer import Trainer
import torch.nn.functional as F
from utils.losses import abCE_loss, CE_loss, consistency_weight
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume, site):
    torch.manual_seed(42)
    train_logger = Logger()

    # DATA LOADERS
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
    supervised_loader = dataloaders.Prostate(site, config['train_supervised'])
    unsupervised_loader = dataloaders.Prostate(site, config['train_unsupervised'])
    val_loader = dataloaders.Prostate(site, config['val_loader'])
    iter_per_epoch = len(unsupervised_loader)
    l = iter(supervised_loader)

    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss
    else:
        sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch, epochs=config['trainer']['epochs'],
                             num_classes=val_loader.dataset.num_classes)

    # MODEL
    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs'])
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(unsupervised_loader),
                                      rampup_ends=rampup_ends)
    # Models
    Shared_Encoder = models.UNet.Encoder()
    model = models.UNet.CCT_Unet(Shared_Encoder, num_classes=val_loader.dataset.num_classes, conf=config['model'],
                       sup_loss=sup_loss, cons_w_unsup=cons_w_unsup,
                       weakly_loss_w=config['weakly_loss_w'], use_weak_lables=config['use_weak_lables'])
    model.float()
    print(f'\n{model}\n')

    # TRAINING
    trainer = Trainer(
        model=model,
        resume=resume,
        config=config,
        supervised_loader=supervised_loader,
        unsupervised_loader=unsupervised_loader,
        val_loader=val_loader,
        iter_per_epoch=iter_per_epoch,
        train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config.json', type=str,
                        help='Path to the config file')
    parser.add_argument('-s', '--site', default='BIDMC', type=str,
                        help='site to train')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    config = json.load(open(args.config))
    # for performance on gpu when input size does not vary
    torch.backends.cudnn.benchmark = True
    main(config, args.resume, args.site)
