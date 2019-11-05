"""
@author: Wei Han
Arrange information for complex scenes via dynamic clustering

Notes:
    The flow of data is quite complex. It includes
        - feeding all data into encoder for clustering,
        - and taking clusters as data for localized tasks,
        - and batches for encoder update
"""

import numpy as np
import torch
import config as cf
import copy

import torchvision
import torchvision.transforms as transforms

import os
import sys
from sklearn.preprocessing import OneHotEncoder

from PIL import Image
import os.path
import pickle
import torch.utils.data as data

global trainset, testset, validset, encoded_trainset, encoded_testset, encoded_validset


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool-batches-py, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True, valid=False, encoded=False, with_encoded=False,
                 classes=np.arange(100), transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.valid = valid
        self.encoded = encoded
        self.with_encoded = with_encoded

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.train_labels = np.array(self.train_labels).astype(np.int64)
            if self.with_encoded:
                self.encoded_train_data = torch.load(self.root + '/train_encoded_data.pkl').astype(np.float32)
                self.encoded_train_labels = torch.load(self.root + '/train_labels.pkl').astype(np.int64)
            else:
                if self.encoded:
                    self.train_data = torch.load(self.root + '/train_encoded_data.pkl').astype(np.float32)
                    self.train_labels = torch.load(self.root + '/train_labels.pkl').astype(np.int64)

            same = classes == np.unique(self.train_labels)
            same = same if isinstance(same, bool) else same.all()
            if not same:
                self.gather_classes(classes, train=True)

            if self.valid:
                if self.with_encoded:
                    labels, class_idx = np.unique(self.train_labels, return_inverse=True)

                    # Sample 20% data as validation set (each label)
                    temp_train_data = self.train_data
                    temp_train_labels = self.train_labels
                    temp_encoded_train_data = self.train_data
                    temp_encoded_train_labels = self.train_labels

                    self.train_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
                    self.valid_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
                    self.encoded_train_data = np.empty((0, 256, 6, 6)).astype(np.float32)
                    self.encoded_valid_data = np.empty((0, 256, 6, 6)).astype(np.float32)

                    self.train_labels = np.empty((0,)).astype(np.int64)
                    self.valid_labels = np.empty((0,)).astype(np.int64)
                    self.encoded_train_labels = np.empty((0,)).astype(np.int64)
                    self.encoded_valid_labels = np.empty((0,)).astype(np.int64)

                    for label in labels:
                        num_class = sum((class_idx == label).astype(int))
                        self.train_data = np.vstack((self.train_data, temp_train_data[class_idx == label][int(num_class * 0.2):, :, :, :]))
                        self.train_labels = np.hstack((self.train_labels, temp_train_labels[class_idx == label][int(num_class * 0.2):]))
                        self.valid_data = np.vstack((self.valid_data, temp_train_data[class_idx == label][:int(num_class * 0.2), :, :, :]))
                        self.valid_labels = np.hstack((self.valid_labels, temp_train_labels[class_idx == label][:int(num_class * 0.2)]))

                        self.encoded_train_data = np.vstack((self.encoded_train_data, temp_encoded_train_data[class_idx == label][int(num_class * 0.2):, :, :, :]))
                        self.encoded_train_labels = np.hstack((self.encoded_train_labels, temp_encoded_train_labels[class_idx == label][int(num_class * 0.2):]))
                        self.encoded_valid_data = np.vstack((self.encoded_valid_data, temp_encoded_train_data[class_idx == label][:int(num_class * 0.2), :, :, :]))
                        self.encoded_valid_labels = np.hstack((self.encoded_valid_labels, temp_encoded_train_labels[class_idx == label][:int(num_class * 0.2)]))

                else:
                    labels, class_idx = np.unique(self.train_labels, return_inverse=True)

                    # Sample 20% data as validation set (each label)
                    temp_train_data = self.train_data
                    temp_train_labels = self.train_labels
                    if not self.encoded:
                        self.train_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
                        self.valid_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
                    else:
                        self.train_data = np.empty((0, 256, 6, 6)).astype(np.float32)
                        self.valid_data = np.empty((0, 256, 6, 6)).astype(np.float32)
                    self.train_labels = np.empty((0,)).astype(np.int64)
                    self.valid_labels = np.empty((0,)).astype(np.int64)

                    for label in labels:
                        num_class = sum((class_idx == label).astype(int))
                        self.train_data = np.vstack((self.train_data, temp_train_data[class_idx == label][int(num_class * 0.2):, :, :, :]))
                        self.train_labels = np.hstack((self.train_labels, temp_train_labels[class_idx == label][int(num_class * 0.2):]))
                        self.valid_data = np.vstack((self.valid_data, temp_train_data[class_idx == label][:int(num_class * 0.2), :, :, :]))
                        self.valid_labels = np.hstack((self.valid_labels, temp_train_labels[class_idx == label][:int(num_class * 0.2)]))

        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.test_labels = np.array(self.test_labels).astype(np.int64)
            if with_encoded:
                self.encoded_test_data = torch.load(self.root + '/test_encoded_data.pkl').astype(np.float32)
                self.encoded_test_labels = torch.load(self.root + '/test_labels.pkl').astype(np.int64)
            else:
                if self.encoded:
                    self.test_data = torch.load(self.root + '/test_encoded_data.pkl').astype(np.float32)
                    self.test_labels = torch.load(self.root + '/test_labels.pkl').astype(np.int64)

            same = classes == np.unique(self.test_labels)
            same = same if isinstance(same, bool) else same.all()
            if not same:
                self.gather_classes(classes, train=False)

    def gather_classes(self, classes, train=True):
        if self.with_encoded:
            if train:
                train_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
                encoded_train_data = np.empty((0, 256, 6, 6)).astype(np.float32)
                if self.train_labels.ndim == 2:
                    all_train_labels = np.argmax(self.train_labels, 1)
                    train_labels = np.empty((0, self.train_labels.shape[1])).astype(np.int64)
                    encoded_train_labels = np.empty((0, self.encoded_train_labels.shape[1])).astype(np.int64)

                    for class_label in classes:
                        train_data = np.vstack((train_data, self.train_data[all_train_labels == class_label]))
                        train_labels = np.vstack((train_labels, self.train_labels[all_train_labels == class_label]))

                        encoded_train_data = np.vstack((encoded_train_data, self.encoded_train_data[all_train_labels == class_label]))
                        encoded_train_labels = np.vstack((encoded_train_labels, self.encoded_train_labels[all_train_labels == class_label]))
                else:
                    all_train_labels = self.train_labels
                    train_labels = np.empty((0,)).astype(np.int64)
                    encoded_train_labels = np.empty((0,)).astype(np.int64)

                    for class_label in classes:
                        train_data = np.vstack((train_data, self.train_data[all_train_labels == class_label]))
                        train_labels = np.hstack((train_labels, self.train_labels[all_train_labels == class_label]))

                        encoded_train_data = np.vstack((encoded_train_data, self.encoded_train_data[all_train_labels == class_label]))
                        encoded_train_labels = np.hstack((encoded_train_labels, self.encoded_train_labels[all_train_labels == class_label]))

                self.train_data = train_data
                self.train_labels = train_labels

                self.train_data = encoded_train_data
                self.train_labels = encoded_train_labels

            else:
                test_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
                encoded_test_data = np.empty((0, 256, 6, 6)).astype(np.float32)
                if self.test_labels.ndim == 2:
                    all_test_labels = np.argmax(self.test_labels, 1)
                    test_labels = np.empty((0, self.test_labels.shape[1])).astype(np.int64)
                    encoded_test_labels = np.empty((0, self.test_labels.shape[1])).astype(np.int64)
                    for class_label in classes:
                        test_data = np.vstack((test_data, self.test_data[all_test_labels == class_label]))
                        test_labels = np.vstack((test_labels, self.test_labels[all_test_labels == class_label]))

                        encoded_test_data = np.vstack((encoded_test_data, self.encoded_test_data[all_test_labels == class_label]))
                        encoded_test_labels = np.vstack((encoded_test_labels, self.encoded_test_labels[all_test_labels == class_label]))
                else:
                    all_test_labels = self.test_labels
                    test_labels = np.empty((0,)).astype(np.int64)
                    encoded_test_labels = np.empty((0,)).astype(np.int64)
                    for class_label in classes:
                        test_data = np.vstack((test_data, self.test_data[all_test_labels == class_label]))
                        test_labels = np.hstack((test_labels, self.test_labels[all_test_labels == class_label]))

                        encoded_test_data = np.vstack((encoded_test_data, self.encoded_test_data[all_test_labels == class_label]))
                        encoded_test_labels = np.hstack((encoded_test_labels, self.encoded_test_labels[all_test_labels == class_label]))

                self.test_data = test_data
                self.test_labels = test_labels

                self.encoded_test_data = encoded_test_data
                self.encoded_test_labels = encoded_test_labels
        else:
            if train:
                if not self.encoded:
                    train_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
                else:
                    train_data = np.empty((0, 256, 6, 6)).astype(np.float32)
                if self.train_labels.ndim == 2:
                    all_train_labels = np.argmax(self.train_labels, 1)
                    train_labels = np.empty((0, self.train_labels.shape[1])).astype(np.int64)
                    for class_label in classes:
                        train_data = np.vstack((train_data, self.train_data[all_train_labels == class_label]))
                        train_labels = np.vstack((train_labels, self.train_labels[all_train_labels == class_label]))
                else:
                    all_train_labels = self.train_labels
                    train_labels = np.empty((0,)).astype(np.int64)
                    for class_label in classes:
                        train_data = np.vstack((train_data, self.train_data[all_train_labels == class_label]))
                        train_labels = np.hstack((train_labels, self.train_labels[all_train_labels == class_label]))
                self.train_data = train_data
                self.train_labels = train_labels
            else:
                if not self.encoded:
                    test_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
                else:
                    test_data = np.empty((0, 256, 6, 6)).astype(np.float32)
                if self.test_labels.ndim == 2:
                    all_test_labels = np.argmax(self.test_labels, 1)
                    test_labels = np.empty((0, self.test_labels.shape[1])).astype(np.int64)
                    for class_label in classes:
                        test_data = np.vstack((test_data, self.test_data[all_test_labels == class_label]))
                        test_labels = np.vstack((test_labels, self.test_labels[all_test_labels == class_label]))
                else:
                    all_test_labels = self.test_labels
                    test_labels = np.empty((0,)).astype(np.int64)
                    for class_label in classes:
                        test_data = np.vstack((test_data, self.test_data[all_test_labels == class_label]))
                        test_labels = np.hstack((test_labels, self.test_labels[all_test_labels == class_label]))

                self.test_data = test_data
                self.test_labels = test_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
            encoded_img, encoded_target = self.encoded_train_data[index], self.encoded_train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
            encoded_img, encoded_target = self.encoded_test_data[index], self.encoded_test_labels[index]

        if self.with_encoded:
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
                encoded_target = self.target_transform(encoded_target)

            encoded_img = torch.from_numpy(encoded_img)
            encoded_img = encoded_img.view(-1)
            encoded_img = torch.sigmoid(encoded_img)

            img = (img, encoded_img)
            target = (target, encoded_target)
        else:
            if not self.encoded:
                # doing this so that it is consistent with all other datasets
                # to return a PIL Image
                img = Image.fromarray(img)

                if self.transform is not None:
                    img = self.transform(img)

                if self.target_transform is not None:
                    target = self.target_transform(target)
            else:
                img = torch.from_numpy(img)
                img = img.view(-1)
                img = torch.sigmoid(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

class Validset():
    def __init__(self, trainset, encoded=False, with_encoded=False):
        self.trainset = trainset
        self.train_data = trainset.valid_data
        self.train_labels = trainset.valid_labels
        self.encoded = encoded
        self.with_encoded = with_encoded
        if with_encoded:
            self.encoded_train_data = trainset.encoded_valid_data
            self.encoded_train_labels = trainset.encoded_valid_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        encoded_img, encoded_target = self.encoded_train_data[index], self.encoded_train_labels[index]

        if self.with_encoded:
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
                encoded_target = self.target_transform(encoded_target)

            encoded_img = torch.from_numpy(encoded_img)
            encoded_img = encoded_img.view(-1)
            encoded_img = torch.sigmoid(encoded_img)

            img = (img, encoded_img)
            target = (target, encoded_target)
        else:
            if not self.encoded:
                # doing this so that it is consistent with all other datasets
                # to return a PIL Image
                img = Image.fromarray(img)

                if self.transform is not None:
                    img = self.transform(img)

                if self.target_transform is not None:
                    target = self.target_transform(target)
            else:
                img = torch.from_numpy(img)
                img = img.view(-1)
                img = torch.sigmoid(img)


        return img, target

    def __len__(self):
        return len(self.train_data)

    def gather_classes(self, classes, train=True):
        if self.with_encoded:
            train_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
            encoded_train_data = np.empty((0, 256, 6, 6)).astype(np.float32)
            if self.train_labels.ndim == 2:
                all_train_labels = np.argmax(self.train_labels, 1)
                train_labels = np.empty((0, self.train_labels.shape[1])).astype(np.int64)
                encoded_train_labels = np.empty((0, self.encoded_train_labels.shape[1])).astype(np.int64)

                for class_label in classes:
                    train_data = np.vstack((train_data, self.train_data[all_train_labels == class_label]))
                    train_labels = np.vstack((train_labels, self.train_labels[all_train_labels == class_label]))

                    encoded_train_data = np.vstack(
                        (encoded_train_data, self.encoded_train_data[all_train_labels == class_label]))
                    encoded_train_labels = np.vstack(
                        (encoded_train_labels, self.encoded_train_labels[all_train_labels == class_label]))
            else:
                all_train_labels = self.train_labels
                train_labels = np.empty((0,)).astype(np.int64)
                encoded_train_labels = np.empty((0,)).astype(np.int64)

                for class_label in classes:
                    train_data = np.vstack((train_data, self.train_data[all_train_labels == class_label]))
                    train_labels = np.hstack((train_labels, self.train_labels[all_train_labels == class_label]))

                    encoded_train_data = np.vstack(
                        (encoded_train_data, self.encoded_train_data[all_train_labels == class_label]))
                    encoded_train_labels = np.hstack(
                        (encoded_train_labels, self.encoded_train_labels[all_train_labels == class_label]))

            self.train_data = train_data
            self.train_labels = train_labels

            self.train_data = encoded_train_data
            self.train_labels = encoded_train_labels

        else:
            if not self.encoded:
                train_data = np.empty((0, 32, 32, 3)).astype(np.uint8)
            else:
                train_data = np.empty((0, 256, 6, 6)).astype(np.float32)
            if self.train_labels.ndim == 2:
                all_train_labels = np.argmax(self.train_labels, 1)
                train_labels = np.empty((0, self.train_labels.shape[1])).astype(np.int64)
                for class_label in classes:
                    train_data = np.vstack((train_data, self.train_data[all_train_labels == class_label]))
                    train_labels = np.vstack((train_labels, self.train_labels[all_train_labels == class_label]))
            else:
                all_train_labels = self.train_labels
                train_labels = np.empty((0,)).astype(np.int64)
                for class_label in classes:
                    train_data = np.vstack((train_data, self.train_data[all_train_labels == class_label]))
                    train_labels = np.hstack((train_labels, self.train_labels[all_train_labels == class_label]))
            self.train_data = train_data
            self.train_labels = train_labels

'''
class Encoded_CIFAR10(CIFAR10):
    def __init__(self, root, train=True, valid=False, classes=np.arange(100), transform=None, target_transform=None):
        super(Encoded_CIFAR10, self).__init__(root, train=train, valid=valid, classes=classes, transform=transform, target_transform=target_transform)
        raise NotImplementedError
        self.train_data = None # Not Implemented
        self.test_data = None # Not Implemented

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        img = torch.from_numpy(img)

        return img, target


class Encoded_CIFAR100(CIFAR100):
    def __init__(self, root, train=True, valid=False, classes=np.arange(100), transform=None, target_transform=None):
        super(Encoded_CIFAR100, self).__init__(root, train=train, valid=valid, encoded=True, classes=classes, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        img = torch.from_numpy(img)

        return img, target
'''

class Dataloders():
    def __init__(self, args, valid=False, one_hot=True):
        self.args = args
        self.valid = valid
        self.one_hot = one_hot

        print('\nData Preparation')
        # Data Uplaod
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        encoded_data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

        # root_path = '/Users/changgang/Documents/DATA/Data For Research/CIFAR'
        root_path = '/HDD/personal/zhengchanggang/CIFAR'
        # root_path = '../data'
        if (args.dataset == 'cifar-10'):
            self.trainset = CIFAR10(root=root_path, train=True, valid=valid, classes=np.arange(10), transform=data_transform)
            self.testset = CIFAR10(root=root_path, train=False, classes=np.arange(10), transform=data_transform)
            self.encoded_trainset = CIFAR10(root=root_path, train=True, valid=valid, encoded=True, classes=np.arange(10),transform=encoded_data_transform)
            self.encoded_testset = CIFAR10(root=root_path, train=False, encoded=True, classes=np.arange(10),transform=encoded_data_transform)
            self.with_encoded_trainset = CIFAR10(root=root_path, train=True, valid=valid, encoded=True, with_encoded=True, classes=np.arange(10),transform=encoded_data_transform)
            self.with_encoded_testset = CIFAR10(root=root_path, train=False, encoded=True, with_encoded=True, classes=np.arange(10),transform=encoded_data_transform)
        else:
            assert args.dataset == 'cifar-100'
            self.trainset = CIFAR100(root=root_path, train=True, valid=valid, classes=np.arange(100),transform=data_transform)
            self.testset = CIFAR100(root=root_path, train=False, classes=np.arange(100), transform=data_transform)
            self.encoded_trainset = CIFAR100(root=root_path, train=True, valid=valid, encoded=True, classes=np.arange(100), transform=encoded_data_transform)
            self.encoded_testset = CIFAR100(root=root_path, train=False, encoded=True, classes=np.arange(100), transform=encoded_data_transform)
            self.with_encoded_trainset = CIFAR100(root=root_path, train=True, valid=valid, encoded=True, with_encoded=True, classes=np.arange(100), transform=encoded_data_transform)
            self.with_encoded_testset = CIFAR100(root=root_path, train=False, encoded=True, with_encoded=True, classes=np.arange(100), transform=encoded_data_transform)

        self.label_transformer = OneHotEncoder(sparse=False, categories='auto').fit(np.array(self.trainset.train_labels).reshape(-1, 1))
        if self.one_hot:  # DEFAULT True
            self.trainset.train_labels = self.label_transformer.transform(np.array(self.trainaset.train_labels).reshape(-1, 1))
            self.testset.test_labels = self.label_transformer.transform(np.array(self.testset.test_labels).reshape(-1, 1))
            self.encoded_trainset.train_labels = self.label_transformer.transform(np.array(self.encoded_trainset.train_labels).reshape(-1, 1))
            self.encoded_testset.test_labels = self.label_transformer.transform(np.array(self.encoded_testset.test_labels).reshape(-1, 1))

            self.with_encoded_trainset.train_labels = self.label_transformer.transform(np.array(self.with_encoded_trainset.train_labels).reshape(-1, 1))
            self.with_encoded_testset.test_labels = self.label_transformer.transform(np.array(self.with_encoded_testset.test_labels).reshape(-1, 1))
            self.with_encoded_trainset.encoded_train_labels = self.label_transformer.transform(np.array(self.with_encoded_trainset.encoded_train_labels).reshape(-1, 1))
            self.with_encoded_testset.encoded_test_labels = self.label_transformer.transform(np.array(self.with_encoded_testset.encoded_test_labels).reshape(-1, 1))


        if valid:
            self.validset = Validset(copy.deepcopy(trainset))
            self.encoded_validset = Validset(copy.deepcopy(self.encoded_trainset), encoded=True)
            self.with_encoded_validset = Validset(copy.deepcopy(self.with_encoded_trainset), encoded=True, with_encoded=True)

        else:
            self.validloader = None
            self.encoded_validloader = None
            self.with_encoded_validset = None

    def get_dataLoder(self, classes, mode='Train', encoded=False, with_encoded=False, one_hot=False, one_hot_based_all=False):
        global trainset, testset, validset, encoded_trainset, encoded_testset, encoded_validset

        # Data Uplaod
        if mode == 'Train':
            train = True
            batch_size = self.args.train_batch_size
            if with_encoded:
                required_set = copy.deepcopy(self.with_encoded_trainset)
            elif encoded:
                required_set = copy.deepcopy(self.encoded_trainset)
            else:
                required_set = copy.deepcopy(self.trainset)
        elif mode == 'Test':
            train = False
            batch_size = 128
            if with_encoded:
                required_set = copy.deepcopy(self.with_encoded_testset)
            elif encoded:
                required_set = copy.deepcopy(self.encoded_testset)
            else:
                required_set = copy.deepcopy(self.testset)
        else:
            assert mode == 'Valid'
            train = True
            batch_size = 128
            if with_encoded:
                required_set = copy.deepcopy(self.with_encoded_validset)
            elif encoded:
                required_set = copy.deepcopy(self.encoded_validset)
            else:
                required_set = copy.deepcopy(self.validset)
        required_set.gather_classes(classes, train=train)
        required_loader = torch.utils.data.DataLoader(required_set, batch_size=batch_size, shuffle=True, num_workers=0)


        if not self.one_hot:
            if one_hot_based_all:
                if train:
                    required_loader.dataset.train_labels = self.label_transformer.transform(np.array(required_loader.dataset.train_labels).reshape(-1, 1))
                    if with_encoded: required_loader.dataset.encoded_train_labels = self.label_transformer.transform(np.array(required_loader.dataset.encoded_train_labels).reshape(-1, 1))
                else:
                    required_loader.dataset.test_labels = self.label_transformer.transform(np.array(required_loader.dataset.test_labels).reshape(-1, 1))
                    if with_encoded: required_loader.dataset.encoded_test_labels = self.label_transformer.transform(np.array(required_loader.dataset.encoded_test_labels).reshape(-1, 1))
            elif one_hot:
                if train:
                    label_transformer = OneHotEncoder(sparse=False, categories='auto').fit(np.array(required_set.train_labels).reshape(-1, 1))
                    required_loader.dataset.train_labels = label_transformer.transform(required_loader.dataset.train_labels.reshape(-1, 1))
                    if with_encoded: required_loader.dataset.encoded_train_labels = label_transformer.transform(required_loader.dataset.encoded_train_labels.reshape(-1, 1))
                else:
                    label_transformer = OneHotEncoder(sparse=False, categories='auto').fit(np.array(required_set.test_labels).reshape(-1, 1))
                    required_loader.dataset.test_labels = label_transformer.transform(required_loader.dataset.test_labels.reshape(-1, 1))
                    if with_encoded: required_loader.dataset.encoded_test_labels = label_transformer.transform(required_loader.dataset.encoded_test_labels.reshape(-1, 1))
            else:
                pass
        else:
            if one_hot_based_all:
                pass
            elif one_hot:
                if train:
                    required_loader.dataset.train_labels = required_loader.dataset.train_labels[:, classes]
                    if with_encoded: required_loader.dataset.encoded_train_labels = required_loader.dataset.encoded_train_labels[:, classes]
                else:
                    required_loader.dataset.test_labels = required_loader.dataset.test_labels[:, classes]
                    if with_encoded: required_loader.dataset.encoded_test_labels = required_loader.dataset.encoded_test_labels[:, classes]
            else:
                if train:
                    required_loader.dataset.train_labels = np.argmax(required_loader.dataset.train_labels, 1)
                    if with_encoded: required_loader.dataset.encoded_train_labels = np.argmax(required_loader.dataset.encoded_train_labels, 1)
                else:
                    required_loader.dataset.test_labels = np.argmax(required_loader.dataset.test_labels, 1)
                    if with_encoded: required_loader.dataset.encoded_test_labels = np.argmax(required_loader.dataset.encoded_test_labels, 1)

        return required_loader