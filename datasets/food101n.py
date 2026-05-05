import os
from .lt_data_food101n import LT_Dataset
import numpy as np


class Food_101N(LT_Dataset):
    classnames_txt = "/root/dataset/Food-101N_release/meta/classes.txt"
    train_txt = "/root/dataset/Food-101N_release/meta/verified_train.tsv"
    test_txt = "/root/dataset/Food-101N_release/meta/verified_val.tsv"

    def __init__(self, root, imb_factor=None, train=True, transform=None):
        super().__init__(root, imb_factor=imb_factor, train=train, transform=transform)

        self.classnames = self.read_classnames()
        self.many_idxs = (np.array(self.cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(self.cls_num_list) >= 20) & (np.array(self.cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(self.cls_num_list) < 20).nonzero()[0]
        # self.names = []
        # with open(self.txt) as f:
        #     for line in f:
        #         self.names.append(self.classnames[int(line.split()[1])])

    def __getitem__(self, index):
        if isinstance(self.transform, list):
            image1, image2, label = super().__getitem__(index)
            # name = self.names[index]
            return image1, image2, label, index, label
        else:
            image, label = super().__getitem__(index)
            # name = self.names[index]
            return image, label

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip().split(" ")
                # folder = line[0]
                # classname = " ".join(line[1:])
                classnames.append(line[0])
        return classnames


class FOOD101N_IR20(Food_101N):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.05, train=train, transform=transform)

class FOOD101N_IR50(Food_101N):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.02, train=train, transform=transform)

class FOOD101N_IR100(Food_101N):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, train=train, transform=transform)
