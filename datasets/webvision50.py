import os
from .lt_data_webvision import LT_Dataset
import numpy as np


class Webvision(LT_Dataset):
    classnames_txt = "/root/projects/win-linux2/LIFT/LIFT-main/datasets/webvision-50/info/queries_google.txt"
    train_txt = "/root/projects/win-linux2/LIFT/LIFT-main/datasets/webvision-50/info/train_filelist_google.txt"
    # test_txt = "/root/projects/win-linux2/LIFT/LIFT-main/datasets/webvision-50/info/val_ILSVRC2012_filelist.txt"
    test_txt = "/root/projects/win-linux2/LIFT/LIFT-main/datasets/webvision-50/info/val_filelist.txt"

    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train, transform)

        self.many_idxs = (np.array(self.cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(self.cls_num_list) >= 20) & (np.array(self.cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(self.cls_num_list) < 20).nonzero()[0]
        # self.classnames = self.read_classnames()
        self.classnames = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
                      "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
                      "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
                      "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
                      "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
                      "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
                      "box turtle", "banded gecko", "green iguana", "Carolina anole",
                      "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
                      "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile"]

        self.names = []
        with open(self.txt) as f:
            for line in f:
                self.names.append(self.classnames[int(line.split()[1])])

    def __getitem__(self, index):
        if isinstance(self.transform, list):
            image1, image2, label, index = super().__getitem__(index)
            # name = self.names[index]
            return image1, image2, label, index, label
        else:
            image, label = super().__getitem__(index)
            name = self.names[index]
            return image, label, name

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames.append(classname)
        return classnames
