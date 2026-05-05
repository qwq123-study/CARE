from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
import re
import torch


class LT_Dataset(Dataset):
    train_txt = ""
    test_txt = ""

    def __init__(self, root, train=True, transform=None, imb_factor=None, noise_ratio=None, random_seed=0):
        self.img_path = []
        self.labels = []
        self.train = train
        self.transform = transform
        self.root = root
        self.cls_num = 100

        self.train_imgs = []
        self.train_labels = {}
        self.clean_labels = {}
        self.noisy_imgs = []
        self.noise_labels = {}
        self.val_imgs = []
        self.val_labels = {}
        control_label_path = self.root + '/split'
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        with open('%s/blue_noise_nl_0.0'%control_label_path, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = self.root + '/all_images' + '/' +entry[0]
                self.train_imgs.append(img_path)
                self.clean_labels[img_path] = int(entry[1])

        with open('%s/red_noise_nl_%.1f'%(control_label_path,0.8),'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                if re.match('^n.*',entry[0]) is None:
                    img_path = self.root + '/all_images' + '/' +entry[0]
                    self.noisy_imgs.append(img_path)
                    self.noise_labels[img_path] = int(entry[1])
        random.shuffle(self.noisy_imgs)
        with open('%s/clean_validation' % control_label_path, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()

                img_path = self.root + '/validation_all' + '/' + entry[0]

                self.val_imgs.append(img_path)
                self.val_labels[img_path] = int(entry[1])
        img_num_list = self.get_img_num_per_cls(len(self.train_imgs) / self.cls_num, self.cls_num, imb_factor, 0)
        self.train_imgs = self.sample_dataset(self.train_imgs, self.clean_labels, img_num_list, self.cls_num, 'select')
        self.data_num = sum(img_num_list)
        select_noisy_num = int(self.data_num / (1 - noise_ratio) - self.data_num)
        self.train_imgs.extend(self.noisy_imgs[:select_noisy_num])
        self.train_labels.update(self.clean_labels)
        self.train_labels.update(self.noise_labels)
        self.data_num = len(self.train_imgs)

        if train:
            self.img_path = self.train_imgs
            for i in range(len(self.train_imgs)):
                self.labels.append(self.train_labels[self.train_imgs[i]])
        else:
            self.img_path = self.val_imgs
            for i in range(len(self.val_imgs)):
                self.labels.append(self.val_labels[self.val_imgs[i]])

        # with open(self.txt) as f:
        #     for line in f:
        #         self.img_path.append(os.path.join(self.root, line.split()[0]))
        #         self.labels.append(int(line.split()[1]))
        self.img_num_list = img_num_list
        self.cls_num_list = self.get_cls_num_list()
        self.clean_cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        self.num_samples = len(self.train_imgs)
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
            return img1, img2, label
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

    def get_img_num_per_cls(self, img_num, cls_num, imb_factor=None, num_meta=None):
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
        img_max = img_num
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

    def sample_dataset(self, train_data, train_label, img_num_list, num_classes, kind):
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
            data_list[j] = [i for i in train_data if train_label[i] == j]

        idx_to_del = []
        for cls_idx, img_id_list in data_list.items():
            '''
            cls_idx : class index
            img_id_list:sample global index list
            data_list:{'cls_idx':[img_id_list],}
            '''
            np.random.shuffle(img_id_list)
            # print(img_id_list)
            img_num = img_num_list[int(cls_idx)]
            # print(img_num)
            if kind == 'delete':
                idx_to_del.extend(img_id_list[:img_num])
            else:
                idx_to_del.extend(img_id_list[img_num:])
        train_data_ = list(set(train_data).difference(set(idx_to_del)))

        return train_data_
