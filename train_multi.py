import os
import json
import argparse
import torch
import dataloaders
import models
import math
from utils import Logger
from trainer_multi import Multi_Trainer
import torch.nn.functional as F
from utils.losses import abCE_loss, CE_loss, consistency_weight
import copy
import matplotlib.pyplot as plt


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def load_sites(config):
    # config
    config_BIDMC = copy.deepcopy(config)
    config_HK = copy.deepcopy(config)
    config_I2CVB = copy.deepcopy(config)
    config_ISBI = copy.deepcopy(config)
    config_ISBI_15 = copy.deepcopy(config)
    config_UCL = copy.deepcopy(config)
    # load the sites specified

    supervised_loader_BIDMC = dataloaders.Prostate("BIDMC",config_BIDMC['train_supervised'])
    unsupervised_loader_BIDMC= dataloaders.Prostate("BIDMC",config_BIDMC['train_unsupervised'])
    val_loader_BIDMC = dataloaders.Prostate("BIDMC",config_BIDMC['val_loader'])
    iter_per_epoch_BIDMC = len(unsupervised_loader_BIDMC)

    supervised_loader_HK = dataloaders.Prostate("HK", config_HK['train_supervised'])
    unsupervised_loader_HK = dataloaders.Prostate("HK", config_HK['train_unsupervised'])
    val_loader_HK = dataloaders.Prostate("HK", config_HK['val_loader'])
    iter_per_epoch_HK = len(unsupervised_loader_HK)

    supervised_loader_I2CVB = dataloaders.Prostate("I2CVB", config_I2CVB['train_supervised'])
    unsupervised_loader_I2CVB = dataloaders.Prostate("I2CVB", config_I2CVB['train_unsupervised'])
    val_loader_I2CVB = dataloaders.Prostate("I2CVB", config_I2CVB['val_loader'])
    iter_per_epoch_I2CVB = len(unsupervised_loader_I2CVB)

    supervised_loader_ISBI = dataloaders.Prostate("ISBI", config_ISBI['train_supervised'])
    unsupervised_loader_ISBI = dataloaders.Prostate("ISBI", config_ISBI['train_unsupervised'])
    val_loader_ISBI = dataloaders.Prostate("ISBI", config_ISBI['val_loader'])
    iter_per_epoch_ISBI = len(unsupervised_loader_ISBI)

    supervised_loader_ISBI_15 = dataloaders.Prostate("ISBI_15", config_ISBI_15['train_supervised'])
    unsupervised_loader_ISBI_15 = dataloaders.Prostate("ISBI_15", config_ISBI_15['train_unsupervised'])
    val_loader_ISBI_15 = dataloaders.Prostate("ISBI_15", config_ISBI_15['val_loader'])
    iter_per_epoch_ISBI_15 = len(unsupervised_loader_ISBI_15)

    supervised_loader_UCL = dataloaders.Prostate("UCL", config_UCL['train_supervised'])
    unsupervised_loader_UCL = dataloaders.Prostate("UCL", config_UCL['train_unsupervised'])
    val_loader_UCL = dataloaders.Prostate("UCL", config_UCL['val_loader'])
    iter_per_epoch_UCL = len(unsupervised_loader_UCL)

    del config_BIDMC, config_HK, config_I2CVB, config_ISBI, config_ISBI_15, config_UCL

    return supervised_loader_BIDMC,unsupervised_loader_BIDMC,val_loader_BIDMC,iter_per_epoch_BIDMC,\
           supervised_loader_HK,unsupervised_loader_HK,val_loader_HK,iter_per_epoch_HK,\
           supervised_loader_I2CVB,unsupervised_loader_I2CVB,val_loader_I2CVB,iter_per_epoch_I2CVB,\
           supervised_loader_ISBI,unsupervised_loader_ISBI,val_loader_ISBI,iter_per_epoch_ISBI,\
           supervised_loader_ISBI_15,unsupervised_loader_ISBI_15,val_loader_ISBI_15,iter_per_epoch_ISBI_15,\
           supervised_loader_UCL,unsupervised_loader_UCL,val_loader_UCL,iter_per_epoch_UCL


def create_model(config, encoder, Num_classes, iter_per_epoch):
    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss
    else:
        sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch, epochs=config['trainer']['epochs'],num_classes = Num_classes)

    # MODEL
    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs'])
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=iter_per_epoch, rampup_ends=rampup_ends)

    model = models.CCT(encoder, num_classes=Num_classes, conf=config['model'],
                       sup_loss=sup_loss, cons_w_unsup=cons_w_unsup,
                       weakly_loss_w=config['weakly_loss_w'], use_weak_lables=config['use_weak_lables'])

    return model.float()


def main(config, resume1, resume2, resume3, resume4, resume5, resume6):
    torch.manual_seed(42)
    # logger for saving
    train_logger_1 = Logger()
    train_logger_2 = Logger()
    train_logger_3 = Logger()
    train_logger_4 = Logger()
    train_logger_5 = Logger()
    train_logger_6 = Logger()

    # DATA LOADERS
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
    supervised_loader_BIDMC, unsupervised_loader_BIDMC, val_loader_BIDMC, iter_per_epoch_BIDMC, \
    supervised_loader_HK, unsupervised_loader_HK, val_loader_HK, iter_per_epoch_HK, \
    supervised_loader_I2CVB, unsupervised_loader_I2CVB, val_loader_I2CVB, iter_per_epoch_I2CVB, \
    supervised_loader_ISBI, unsupervised_loader_ISBI, val_loader_ISBI, iter_per_epoch_ISBI, \
    supervised_loader_ISBI_15, unsupervised_loader_ISBI_15, val_loader_ISBI_15, iter_per_epoch_ISBI_15, \
    supervised_loader_UCL, unsupervised_loader_UCL, val_loader_UCL, iter_per_epoch_UCL = load_sites(config)

    # Models
    Shared_Encoder = models.encoder.Encoder(True)

    model_BIDMC = create_model(config,Shared_Encoder,val_loader_BIDMC.dataset.num_classes,iter_per_epoch_BIDMC)
    model_HK = create_model(config, Shared_Encoder, val_loader_HK.dataset.num_classes, iter_per_epoch_HK)
    model_I2CVB = create_model(config, Shared_Encoder, val_loader_I2CVB.dataset.num_classes, iter_per_epoch_I2CVB)
    model_ISBI = create_model(config, Shared_Encoder, val_loader_ISBI.dataset.num_classes, iter_per_epoch_ISBI)
    model_ISBI_15 = create_model(config, Shared_Encoder, val_loader_ISBI_15.dataset.num_classes, iter_per_epoch_ISBI_15)
    model_UCL = create_model(config, Shared_Encoder, val_loader_UCL.dataset.num_classes, iter_per_epoch_UCL)

    # TRAINING
    trainer = Multi_Trainer(
        model_1=model_BIDMC, model_2=model_HK, model_3=model_I2CVB, model_4=model_ISBI, model_5=model_ISBI_15, model_6=model_UCL,
        resume1=resume1, resume2=resume2, resume3=resume3, resume4=resume4, resume5=resume5, resume6=resume6, config=config,
        supervised_loader_1=supervised_loader_BIDMC,unsupervised_loader_1=unsupervised_loader_BIDMC,iter_per_epoch_1=iter_per_epoch_BIDMC,
        supervised_loader_2=supervised_loader_HK, unsupervised_loader_2=unsupervised_loader_HK,iter_per_epoch_2=iter_per_epoch_HK,
        supervised_loader_3=supervised_loader_I2CVB, unsupervised_loader_3=unsupervised_loader_I2CVB,iter_per_epoch_3=iter_per_epoch_I2CVB,
        supervised_loader_4=supervised_loader_ISBI, unsupervised_loader_4=unsupervised_loader_ISBI,iter_per_epoch_4=iter_per_epoch_ISBI,
        supervised_loader_5=supervised_loader_ISBI_15, unsupervised_loader_5=unsupervised_loader_ISBI_15,iter_per_epoch_5=iter_per_epoch_ISBI_15,
        supervised_loader_6=supervised_loader_UCL, unsupervised_loader_6=unsupervised_loader_UCL,iter_per_epoch_6=iter_per_epoch_UCL,
        val_loader_1=val_loader_BIDMC,val_loader_2=val_loader_HK,val_loader_3=val_loader_I2CVB,val_loader_4=val_loader_ISBI,
        val_loader_5=val_loader_ISBI_15,val_loader_6=val_loader_UCL,train_logger_1=train_logger_1,train_logger_2=train_logger_2,
        train_logger_3=train_logger_3,train_logger_4=train_logger_4,train_logger_5=train_logger_5,train_logger_6=train_logger_6)

    trainer.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config.json',type=str,
                        help='Path to the config file')
    parser.add_argument('-r1', '--resume1', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r2', '--resume2', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r3', '--resume3', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r4', '--resume4', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r5', '--resume5', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r6', '--resume6', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)

    args = parser.parse_args()

    config = json.load(open(args.config))
    # for performance on gpu when input size does not vary
    torch.backends.cudnn.benchmark = True
    main(config, args.resume1, args.resume2, args.resume3, args.resume4, args.resume5, args.resume6)
