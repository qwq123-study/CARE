from .lt_mini_imagenet import LT_Dataset
import numpy as np

class Mini_Imagenet(LT_Dataset):
    classnames_txt = "/mnt/d/2Ddataset/mini-imagenet/class_name.txt"
    train_txt = ""
    test_txt = ""

    def __init__(self, root, train=True, transform=None, imb_factor=None, noise_ratio=None):
        super().__init__(root, train, transform, imb_factor, noise_ratio)

        self.many_idxs = (np.array(self.img_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(self.img_num_list) >= 20) & (np.array(self.cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(self.img_num_list) < 20).nonzero()[0]
        self.classnames = self.read_classnames()
        # print(len(self.classnames))

    def __getitem__(self, index):
        if isinstance(self.transform, list):
            image1, image2, label = super().__getitem__(index)
            return image1, image2, label, index, label
        else:
            image, label = super().__getitem__(index)
            return image, label

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                classname = line
                classnames.append(classname)
        return classnames

class MINI_IMAGENET_IR100_NR60(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.6, train=train, transform=transform)

class MINI_IMAGENET_IR100_NR50(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.5, train=train, transform=transform)

class MINI_IMAGENET_IR100_NR40(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.4, train=train, transform=transform)

class MINI_IMAGENET_IR100_NR30(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.3, train=train, transform=transform)

class MINI_IMAGENET_IR100_NR20(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.2, train=train, transform=transform)

class MINI_IMAGENET_IR100_NR10(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.1, train=train, transform=transform)


class MINI_IMAGENET_IR10_NR60(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.6, train=train, transform=transform)

class MINI_IMAGENET_IR10_NR50(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.5, train=train, transform=transform)

class MINI_IMAGENET_IR10_NR40(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.4, train=train, transform=transform)

class MINI_IMAGENET_IR10_NR30(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.3, train=train, transform=transform)

class MINI_IMAGENET_IR10_NR20(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.2, train=train, transform=transform)

class MINI_IMAGENET_IR10_NR10(Mini_Imagenet):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.1, train=train, transform=transform)
