import json
from collections import defaultdict
import random

import numpy as np
import torchvision
import os
import torch
import copy

from PIL import Image


class cifar100(torchvision.datasets.CIFAR100):
    def __getitem__(self, index):
        image, target, clean_labels = self.data[index], self.targets[index], self.clean_labels[index]
        image = Image.fromarray(image)
        if isinstance(self.transform, list):
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            return img1, img2, target, index, clean_labels
        else:
            image = self.transform(image)
            return image, target


class IMBALANCECIFAR100(cifar100):
    cls_num = 100

    def __init__(self, root, imb_factor=None, noise_ratio=None, noise_mode=None, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

        self.labels = self.targets
        self.clean_labels = self.targets

        if train and imb_factor is not None:
            train_data = self.data
            train_label = self.targets
            img_num_list = self.get_img_num_per_cls_1(self.cls_num, imb_factor, 0)
            train_data, train_label = self.sample_dataset_1(train_data, train_label, img_num_list, self.cls_num, 'select')
            self.data_num = sum(img_num_list)
            self.labels = train_label
            self.clean_labels = train_label
            self.clean_cls_num_list = self.get_cls_num_list()
            self.data = train_data

        self.classnames = self.classes

        if train and noise_ratio is not None:
            if noise_mode == 'unif':
                noisy_transaction_matrix_real = self.uniform_mix_c_1(noise_ratio, self.cls_num)
            elif noise_mode == 'flip':
                noisy_transaction_matrix_real = self.flip_labels_c_1(noise_ratio, self.cls_num)
            noisy_label = copy.deepcopy(train_label)
            for i in range(sum(img_num_list)):
                noisy_label[i] = np.random.choice(self.cls_num, p=noisy_transaction_matrix_real[train_label[i]])
            self.labels = noisy_label
        self.targets = self.labels
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        self.num_samples = len(self.data)
        self.many_idxs = np.arange(0, 36)  # 包含从 0 到 35
        self.med_idxs = np.arange(36, 71)  # 包含从 36 到 70
        self.few_idxs = np.arange(71, 100)  # 包含从 71 到 99
    def sample_dataset_1(self, train_data, train_label, img_num_list, num_classes, kind):
        """
        Args:
            dataset
            img_num_list
            num_classes
            kind
        Returns:


        """
        data_list = {}
        for j in range(num_classes):
            data_list[j] = [i for i, label in enumerate(train_label) if label == j]

        idx_to_del = []
        for cls_idx, img_id_list in data_list.items():
            '''
            cls_idx : class index
            img_id_list:sample global index list
            data_list:{'cls_idx':[img_id_list],}
            '''
            np.random.shuffle(img_id_list)
            img_num = img_num_list[int(cls_idx)]
            if kind == 'delete':
                idx_to_del.extend(img_id_list[:img_num])
            else:
                idx_to_del.extend(img_id_list[img_num:])

        # new_dataset = copy.deepcopy(dataset)
        train_label = np.delete(train_label, idx_to_del, axis=0)
        train_data = np.delete(train_data, idx_to_del, axis=0)
        # data_index = np.delete(data_index, idx_to_del, axis=0)
        return train_data, train_label

    def get_img_num_per_cls_1(self, cls_num, imb_factor=None, num_meta=None):
        """
        Get a list of image numbers for each class, given cifar version
        Num of imgs follows emponential distribution
        img max: 5000 / 500 * e^(-lambda * 0);
        img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
        exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
        args:
          cifar_version: str, '10', '100', '20'
          imb_factor: float, imbalance factor: img_min/img_max,
            None if geting default cifar data number
        output:
          img_num_per_cls: a list of number of images per class
        """

        img_max = 500 - num_meta

        if imb_factor is None:
            return [img_max] * cls_num
        img_num_per_cls = []
        imbalance_ratios = []
        for cls_idx in range(cls_num):
            ratio = imb_factor ** (cls_idx / (cls_num - 1.0))
            imbalance_ratios.append(ratio)
        for cls_idx in range(cls_num):
            ratio = imbalance_ratios[cls_idx]
            num = img_max * ratio
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list

    def uniform_mix_c_1(self, mixing_ratio, num_classes):
        """
        returns a linear interpolation of a uniform matrix and an identity matrix
        """
        return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
            (1 - mixing_ratio) * np.eye(num_classes)

    def flip_labels_c_1(self, corruption_prob, num_classes, seed=1):
        """
        returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
        concentrated in only one other entry for each row
        """
        np.random.seed(seed)
        C = np.eye(num_classes) * (1 - corruption_prob)
        row_indices = np.arange(num_classes)
        for i in range(num_classes):
            C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
        torch.save(C, 'noisy_transaction_matrix_real.pt')
        return C


class CIFAR100(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=None, train=train, transform=transform)


class CIFAR100_IR10(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, train=train, transform=transform)


class CIFAR100_IR50(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.02, train=train, transform=transform)

class CIFAR100_IR100_NR70_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.7, noise_mode='unif', train=train, transform=transform)

class CIFAR100_IR100_NR60_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.6, noise_mode='unif', train=train, transform=transform)

class CIFAR100_IR100_NR50_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.5, noise_mode='unif', train=train, transform=transform)


class CIFAR100_IR100_NR40_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.4, noise_mode='unif', train=train, transform=transform)


class CIFAR100_IR100_NR30_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.3, noise_mode='unif', train=train, transform=transform)


class CIFAR100_IR100_NR20_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.2, noise_mode='unif', train=train, transform=transform)


class CIFAR100_IR100_NR10_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.1, noise_mode='unif', train=train, transform=transform)

class CIFAR100_IR100_NR70_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.7, noise_mode='flip', train=train, transform=transform)

class CIFAR100_IR100_NR60_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.6, noise_mode='flip', train=train, transform=transform)

class CIFAR100_IR100_NR50_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.5, noise_mode='flip', train=train, transform=transform)


class CIFAR100_IR100_NR40_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.4, noise_mode='flip', train=train, transform=transform)


class CIFAR100_IR100_NR30_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.3, noise_mode='flip', train=train, transform=transform)


class CIFAR100_IR100_NR20_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.2, noise_mode='flip', train=train, transform=transform)


class CIFAR100_IR100_NR10_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.1, noise_mode='flip', train=train, transform=transform)

class CIFAR100_IR100(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, train=train, transform=transform)

class CIFAR100_IR10_NR60_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.6, noise_mode='unif', train=train, transform=transform)

class CIFAR100_IR10_NR50_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.5, noise_mode='unif', train=train, transform=transform)


class CIFAR100_IR10_NR40_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.4, noise_mode='unif', train=train, transform=transform)


class CIFAR100_IR10_NR30_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.3, noise_mode='unif', train=train, transform=transform)


class CIFAR100_IR10_NR20_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.2, noise_mode='unif', train=train, transform=transform)


class CIFAR100_IR10_NR10_Symmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.1, noise_mode='unif', train=train, transform=transform)

class CIFAR100_IR10_NR60_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.6, noise_mode='flip', train=train, transform=transform)

class CIFAR100_IR10_NR50_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.5, noise_mode='flip', train=train, transform=transform)


class CIFAR100_IR10_NR40_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.4, noise_mode='flip', train=train, transform=transform)


class CIFAR100_IR10_NR30_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.3, noise_mode='flip', train=train, transform=transform)


class CIFAR100_IR10_NR20_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.2, noise_mode='flip', train=train, transform=transform)


class CIFAR100_IR10_NR10_ASymmetric(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.1, noise_mode='flip', train=train, transform=transform)