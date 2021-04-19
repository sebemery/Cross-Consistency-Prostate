import random, math
import numpy as np
import cv2
import torch
import os
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from math import ceil
import matplotlib.pyplot as plt

class BaseDataSet(Dataset):
    def __init__(self, site, data_dir, split, base_size=None, augment=True, val=False,
                jitter=False, use_weak_lables=False, weak_labels_output=None, scale=False, flip=False, translate=False, rotate=False,
                blur=False, return_id=False, n_labeled_examples=None):

        self.root = data_dir
        self.split = split
        self.mean = None
        self.std = None
        self.augment = augment
        self.jitter = jitter
        self.return_id = return_id
        self.val = val
        self.site = site
        self.slices = None
        self.labels = None

        self.use_weak_lables = use_weak_lables
        self.weak_labels_output = weak_labels_output

        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.translate = translate
            self.rotate = rotate
            self.blur = blur

        self.jitter_tf = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        self.to_tensor = transforms.ToTensor()
        self.normalize = None

        self.files = []
        self._set_files()
        self._load_data_as_slices()

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data_as_slices(self):
        raise NotImplementedError

    def _rotate(self, image, label):
        # Rotate the image with an angle between -10 and 10
        h, w = image.shape
        angle = random.randint(-10, 10)
        center = (w / 2, h / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_CUBIC)#, borderMode=cv2.BORDER_REFLECT)
        label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)
        return image, label

    def _blur(self, image, label):
        # Gaussian Blud (sigma between 0 and 1.5)
        sigma = random.random() * 1.5
        ksize = int(3.3 * sigma)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return image, label

    def _flip(self, image, label):
        # Random H flip
        img = image.copy()
        if random.random() > 0.5:
            img[0, :, :] = np.fliplr(image[0]).copy()
            label = np.fliplr(label).copy()
        return img, label

    def _translate(self, img, label, shift):
        
        if shift == 0:
            return img, label
        else:
            direction = ['right', 'left', 'down', 'up']
            i = random.randint(0, 3)
            img = img.copy()
            lbl = label.copy()
            if direction[i] == 'right':
                right_slice = img[0, :, -shift:].copy()
                img[0, :, shift:] = img[0, :, :-shift]
                right_lbl = lbl[:, -shift:].copy()
                lbl[:, shift:] = lbl[:, :-shift]
            if direction[i] == 'left':
                left_slice = img[0, :, :shift].copy()
                img[0, :, :-shift] = img[0, :, shift:]
                left_lbl = lbl[:, :shift].copy()
                lbl[:, :-shift] = lbl[:, shift:]
            if direction[i] == 'down':
                down_slice = img[0, -shift:, :].copy()
                img[0, shift:, :] = img[0, :-shift, :]
                down_lbl = lbl[-shift:, :].copy()
                lbl[shift:, :] = lbl[:-shift, :]
            if direction[i] == 'up':
                upper_slice = img[0, :shift, :].copy()
                img[0, :-shift, :] = img[0, shift:, :]
                upper_lbl = lbl[-shift:, :].copy()
                lbl[shift:, :] = lbl[:-shift, :]
            return img, lbl

    def _val_augmentation(self, image, label):

        #image = Image.fromarray(np.uint8(image))
        image = torch.from_numpy(image)
        return image.float(), label

    def _augmentation(self, image, label):
        
        if self.translate:
            if random.randint(0, 1) == 1:
                shift_pixel = random.randint(0, 10)
                image, label = self._translate(image, label, shift=shift_pixel)

        if self.flip:
            if random.randint(0, 1) == 1:
                image, label = self._flip(image, label)
        
        #image = Image.fromarray(np.uint8(image))
        # image = self.jitter_tf(image) if self.jitter else image    
        image = torch.from_numpy(image)
        
        return image.float(), label

    def __len__(self):
        return len(self.slices)
  
    def __getitem__(self, index):

        image = self.slices[index]
        label = self.labels[index]
        

        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return image, label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


class testDataset(Dataset):
    def __init__(self, site):
        self.site = site
        # path of the folder
        images_path_files = os.path.join("data/split_prostate/Train-Test-Validation", f"test_files_tr_te_val_{site}" + ".txt")
        # get all path for the image in the directory
        file_list = [line.rstrip().split(',') for line in tuple(open(images_path_files, "r"))]
        self.files, self.labels = list(zip(*file_list))
        self.slices = None
        self.lbls = None
        self.id = []
        self.to_tensor = transforms.ToTensor()
        if site == 'ISBI' or site == 'ISBI_15':
            self.num_classes = 3
        else:
            self.num_classes = 2
        self._load_data_as_slices()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        image = self.slices[index]
        label = self.lbls[index]
        image_id = self.id[index]+f'_{index}'
        image = torch.from_numpy(image)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return image.float(), label, image_id

    def _load_data_as_slices(self):
        # get the data
        image_path = os.path.join(f"data/{self.site}", self.files[0][1:])
        image = nib.load(image_path)
        label_path = os.path.join(f"data/{self.site}", self.labels[0][1:])
        label = nib.load(label_path)
        case = self.files[0][1:]
        case = case[:-4]
        case = case[7:]
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

        slices = image
        labels = label
        for nb_slices in range(image.size(0)):
            self.id.append(case)

        for i in range(1, len(self.files)):
            # get the data
            image_path = os.path.join(f"data/{self.site}", self.files[i][1:])
            image = nib.load(image_path)
            label_path = os.path.join(f"data/{self.site}", self.labels[i][1:])
            label = nib.load(label_path)
            case = self.files[i][1:]
            case = case[:-4]
            case = case[7:]
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
            for nb_slices in range(image.size(0)):
                self.id.append(case)

        #self.mean = slices.mean().item()  # 0.449
        #self.std = slices.std().item()  # 0.226
        #slices = slices.sub(self.mean).div(self.std)
        self.slices = slices.numpy().transpose((0, 2, 1))
        self.lbls = labels.numpy().transpose((0, 2, 1))
        self.slices = np.expand_dims(self.slices, axis=1)

    def get_prostate_slices(self, img, label):
        """
        Return all slices of the volume where there is a prostate segmentation mask which is non zero
        """
        mean = img.mean().item()
        std = img.std().item()
        img = img.sub(mean).div(std)
        indices = []
        for i, mask in enumerate(label):
            # get all non zero value of the segmentation mask
            non_zero = torch.nonzero(mask, as_tuple=False)
            # check if there is non zero value in the seg mask and keep the indices
            if non_zero.size(0) > 0:
                indices.append(i)
                # img[i] = (img[i] - torch.min(img[i])) / (torch.max(img[i]) - torch.min(img[i]))
                # img[i] = (img[i] - mean)/std

        return img[indices], label[indices]