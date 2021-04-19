from base import BaseDataSet, BaseDataLoader
from utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import json
import nibabel as nib
import matplotlib.pyplot as plt
from utils.helpers import DeNormalize

class MRI_dataset(BaseDataSet):
    def __init__(self, site, **kwargs):
        if site == 'ISBI' or site == 'ISBI_15':
            self.num_classes = 3
        else:
            self.num_classes = 2

        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(MRI_dataset, self).__init__(site, **kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, self.site)
        if self.split == "val":
            file_list = os.path.join("data/split_prostate/Train-Test-Validation", f"{self.split}_files_tr_te_val_{self.site}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join("data/split_prostate/Train-Test-Validation", f"{self.split}_files_tr_te_val_{self.site}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split(',') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data_as_slices(self):
        # get the data
        image_path = os.path.join(self.root, self.files[0][1:])
        image = nib.load(image_path)
        label_path = os.path.join(self.root, self.labels[0][1:])
        label = nib.load(label_path)
        #convert to tensor
        image = image.get_fdata()
        image = torch.from_numpy(image)
        label = label.get_fdata()
        label = torch.from_numpy(label)
        # permute dimension of volume
        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)
        # keep slices with prostate region
        image, label = self.get_prostate_slices(image, label)

        slices = image
        labels = label

        for i in range(1, len(self.files)):
            # get the data
            image_path = os.path.join(self.root, self.files[i][1:])
            image = nib.load(image_path)
            label_path = os.path.join(self.root, self.labels[i][1:])
            label = nib.load(label_path)
            # convert to tensor
            image = image.get_fdata()
            image = torch.from_numpy(image)
            label = label.get_fdata()
            label = torch.from_numpy(label)
            # permute dimension of volume
            image = image.permute(2, 0, 1)
            label = label.permute(2, 0, 1)
            # keep slices with prostate region
            image, label = self.get_prostate_slices(image, label)
            slices = torch.cat((slices, image), 0)
            labels = torch.cat((labels, label), 0)

        self.mean = slices.mean().item()
        self.std = slices.std().item()
        # self.normalize = transforms.Normalize(self.mean, self.std)
        # self.denormalize = DeNormalize(self.mean, self.std)
        slices = slices.sub(self.mean).div(self.std)
        self.slices = slices.numpy().transpose((0, 2, 1))
        self.labels = labels.numpy().transpose((0, 2, 1))
        self.slices = np.expand_dims(self.slices, axis=1)
        
    def get_prostate_slices(self, img, label):
        """
        Return all slices of the volume where there is a prostate segmentation mask which is non zero
        """
        indices = []
        for i, mask in enumerate(label):
            # get all non zero value of the segmentation mask
            non_zero = torch.nonzero(mask, as_tuple=False)
            # check if there is non zero value in the seg mask and keep the indice
            if non_zero.size(0) > 0:
                indices.append(i)
                img[i] = (img[i]-torch.min(img[i])) / (torch.max(img[i])-torch.min(img[i]))

        return img[indices], label[indices]


class Prostate(BaseDataLoader):
    def __init__(self, site, kwargs):

        self.batch_size = kwargs.pop('batch_size')
            
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        self.dataset = MRI_dataset(site,**kwargs)
        self.MEAN = self.dataset.mean
        self.STD = self.dataset.std
        super(Prostate, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)