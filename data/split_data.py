import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import random
import argparse
from operator import itemgetter


class split_data_txt :
    def __init__(self, split_tr, split_te, split_val, split_sup, seed):
        # directory of the data
        self.root_dir_BIDMC = 'BIDMC/Images'
        self.seg_dir_BIDMC = 'BIDMC/Segmentation'
        self.root_dir_HK = 'HK/Images'
        self.seg_dir_HK = 'HK/Segmentation'
        self.root_dir_I2CVB = 'I2CVB/Images'
        self.seg_dir_I2CVB = 'I2CVB/Segmentation'
        self.root_dir_ISBI = 'ISBI/Images'
        self.seg_dir_ISBI = 'ISBI/Segmentation'
        self.root_dir_ISBI_15 = 'ISBI_15/Images'
        self.seg_dir_ISBI_15 = 'ISBI_15/Segmentation'
        self.root_dir_UCL = 'UCL/Images'
        self.seg_dir_UCL = 'UCL/Segmentation'
        # split parameters
        self.split_tr = split_tr
        self.split_te = split_te
        self.split_val = split_val
        self.split_sup = split_sup
        # seed
        self.seed = seed

    def read_file_names(self):
        """
        read automatically the six datasets file's name
        """
        files_BIDMC = os.listdir(self.root_dir_BIDMC)
        masks_BIDMC = os.listdir(self.seg_dir_BIDMC)
        files_HK = os.listdir(self.root_dir_HK)
        masks_HK = os.listdir(self.seg_dir_HK)
        files_I2CVB = os.listdir(self.root_dir_I2CVB)
        masks_I2CVB = os.listdir(self.seg_dir_I2CVB)
        files_ISBI = os.listdir(self.root_dir_ISBI)
        masks_ISBI = os.listdir(self.seg_dir_ISBI)
        files_ISBI_15 = os.listdir(self.root_dir_ISBI_15)
        masks_ISBI_15 = os.listdir(self.seg_dir_ISBI_15)
        files_UCL = os.listdir(self.root_dir_UCL)
        masks_UCL = os.listdir(self.seg_dir_UCL)
        site_files = [files_BIDMC, files_HK, files_I2CVB, files_ISBI, files_ISBI_15, files_UCL]
        site_masks = [masks_BIDMC, masks_HK, masks_I2CVB, masks_ISBI, masks_ISBI_15, masks_UCL]
        return site_files, site_masks

    def generate_txt(self, file_name, files, masks):
        # check if the dir exists
        if not os.path.exists('split_prostate'):
            os.makedirs('split_prostate')
        # write txt in the respective dir
        if self.split_te == None:
            if not os.path.exists('split_prostate/Train-Validation'):
                os.makedirs('split_prostate/Train-Validation')
            path_name = os.path.join('split_prostate/Train-Validation', file_name)
        else:
            if not os.path.exists('split_prostate/Train-Test-Validation'):
                os.makedirs('split_prostate/Train-Test-Validation')
            path_name = os.path.join('split_prostate/Train-Test-Validation', file_name)

        with open(path_name, 'w') as f:
            for str1, str2 in zip(files, masks):
                f.write("%s\n" % ('/Images/'+str1 + ',' + '/Segmentation/' + str2))

    def write_txt(self):
        """
        Write in different html files the patient files number which belong to the training, test and validation data
        """
        random.seed(self.seed)
        sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_15', 'UCL']
        site_files, site_masks = self.read_file_names()

        for idx in range(len(site_files)):
            # get files name
            files = site_files[idx]
            segmentation = site_masks[idx]
            len_files = len(files)
            site_label = sites[idx]
            # shuffle the patient's file
            indices = list(range(len_files))
            random.shuffle(indices)

            if self.split_te == None:
                # get list of indices
                split = int(np.floor(self.split_val*len_files))
                train_idx = indices[split:]
                valid_idx = indices[:split]
                split_un_sup = int(self.split_sup * len(train_idx))
                train_sup_idx = train_idx[split_un_sup:]
                train_unsup_idx = train_idx[:split_un_sup]
                # slice the original list with the indices
                files_train_sup = [files[i] for i in train_sup_idx]
                files_train_unsup = [files[i] for i in train_unsup_idx]
                masks_train_sup = [segmentation[i] for i in train_sup_idx]
                masks_train_unsup = [segmentation[i] for i in train_unsup_idx]
                files_val = [files[i] for i in valid_idx]
                masks_val = [segmentation[i] for i in valid_idx]
                # write txt file
                self.generate_txt('train_supervised_files_tr_val_' + site_label + '.txt', files_train_sup, masks_train_sup)
                self.generate_txt('train_unsupervised_files_tr_val_' + site_label + '.txt', files_train_unsup, masks_train_unsup)
                self.generate_txt('val_files_tr_val_' + site_label + '.txt', files_val, masks_val)
            else:
                # get list of indices
                split_tr_te = int(np.floor(self.split_te*len_files))
                train_idx_tmp = indices[split_tr_te:]
                test_idx = indices[:split_tr_te]
                split_tr_val = int(self.split_val*len(train_idx_tmp))
                train_idx = train_idx_tmp[split_tr_val:]
                valid_idx = train_idx_tmp[:split_tr_val]
                split_un_sup = int(self.split_sup * len(train_idx))
                train_sup_idx = train_idx[split_un_sup:]
                train_unsup_idx = train_idx[:split_un_sup]
                # slice the original list with the indices
                files_train_sup = [files[i] for i in train_sup_idx]
                files_train_unsup = [files[i] for i in train_unsup_idx]
                masks_train_sup = [segmentation[i] for i in train_sup_idx]
                masks_train_unsup = [segmentation[i] for i in train_unsup_idx]
                files_test = [files[i] for i in test_idx]
                masks_test = [segmentation[i] for i in test_idx]
                files_val = [files[i] for i in valid_idx]
                masks_val = [segmentation[i] for i in valid_idx]
                # write txt file
                self.generate_txt('train_supervised_files_tr_te_val_' + site_label + '.txt', files_train_sup, masks_train_sup)
                self.generate_txt('train_unsupervised_files_tr_te_val_' + site_label + '.txt', files_train_unsup, masks_train_unsup)
                self.generate_txt('test_files_tr_te_val_'+site_label+'.txt', files_test, masks_test)
                self.generate_txt('val_files_tr_te_val_'+site_label+'.txt', files_val, masks_val)


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='split generation')
    parser.add_argument('-tr', '--training', default=1.0, type=float,
                        help='training data splitting proportion with test')
    parser.add_argument('-te', '--test', default=None, type=float,
                        help='test data splitting proportion with train')
    parser.add_argument('-val', '--validation', default=0.2, type=float,
                        help='validation proportion over the training data data')
    parser.add_argument('-sup', '--supervised', default=0.5, type=float,
                        help='proportion of supervised (labelled) in the training data')
    parser.add_argument('-rand', '--seed', default=10, type=int,
                        help='seed to shuffle the patient list')
    args = parser.parse_args()

    generate_splits = split_data_txt(args.training, args.test, args.validation, args.supervised, args.seed)
    generate_splits.write_txt()
