import os
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class LT_Dataset(Dataset):
    train_txt = ""
    test_txt = ""

    def __init__(self, root, train=True, transform=None, imb_factor=None):
        self.img_path = []
        self.labels = []
        self.train = train
        self.transform = transform
        self.root = root
        self.cls_num = 50
        self.num_samples = 0

        if train:
            self.txt = self.train_txt
        else:
            self.root = os.path.join(self.root, "val_images_256")
            # self.root = os.path.join(self.root, "val")
            self.txt = self.test_txt

        with open(self.txt) as f:
            for line in f:

                # parts = line.split()
                # if len(parts) > 0:
                #     image_name = parts[0].strip()
                #     # 提取文件名（ILSVRC2012_val_xxxxxxxx.JPEG）
                #     filename = os.path.basename(image_name)
                # self.img_path.append(os.path.join(self.root, filename))

                self.img_path.append(os.path.join(self.root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
                if train:
                    self.num_samples += 1

        if train and imb_factor is not None:
            np.random.seed(0)
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            # print(len(self.img_path))
            # print("*"*50)
            # print(len(self.labels))

        self.cls_num_list = self.get_cls_num_list()
        self.clean_cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        self.targets = self.labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        # if self.transform is not None:
        #     image = self.transform(image)
        #
        # return image, label
        if isinstance(self.transform, list):
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            return img1, img2, label, index
        else:
            image = self.transform(image)
            return image, label
    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            selec_idx = selec_idx.tolist()
            # print("selec_idx_length:")
            # print(len(selec_idx))
            # print("the_img_num_length:")
            # print(the_img_num)
            if the_img_num > len(selec_idx):
                the_img_num = len(selec_idx)
            new_data.extend([self.img_path[idx] for idx in selec_idx])
            # new_data.append([self.img_path[idx] for idx in selec_idx])
            # 这行代码的作用是将形成的列表作为一个整体添加到new_data这个列表的末尾
            new_targets.extend([the_class, ] * the_img_num)
        # new_targets = np.array(new_targets, dtype=np.int64).tolist()
        self.img_path = new_data
        self.labels = new_targets


    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = len(self.img_path) / cls_num
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
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
