import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T


class miniImagenet(Dataset):
    """

    re-implement to load data from Spyros Gidaris and Niko Komodakis

"""

    def __init__(self, root, resize, split, mode, augment):

        self.resize = resize
        self.split = split
        self.mode = mode

        mean_pix = [x/255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

        padding = 8  # im_size = 84
        if augment == 0:
            self.transform = T.Compose([
                T.Resize((self.resize + padding, self.resize + padding)),
                T.CenterCrop(self.resize),
                T.ToTensor(),
                T.Normalize(mean=mean_pix, std=std_pix)
            ])
        elif augment == 1:
            self.transform = T.Compose([
                T.Resize((self.resize+padding, self.resize+padding)),
                T.RandomCrop(self.resize),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                T.ToTensor(),
                T.Normalize(mean=mean_pix, std=std_pix)
            ])
        elif augment == 2:
            self.transform = T.Compose([
                T.Resize((self.resize + padding, self.resize + padding)),
                T.TenCrop(self.resize),
                T.Lambda(lambda crops: [T.ToTensor()(crop) for crop in crops]),
                T.Lambda(lambda crops: torch.stack([T.Normalize(mean=mean_pix, std=std_pix)(crop) for crop in crops]))
            ])
        else:
            raise NameError('Augment mode {} not implemented.'.format(augment))

        self.path = os.path.join(root)
        if self.mode == 'regular':
            if self.split == 'train':
                file_name = os.path.join(root, 'train_train')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
            elif self.split == 'val':
                file_name = os.path.join(root, 'train_val')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
            else:  # self.split == 'test'
                file_name = os.path.join(root, 'train_test')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
        elif self.mode == 'openfew':
            if self.split == 'train':
                file_name = os.path.join(root, 'train_train')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
            elif self.split == 'val':
                file_name = os.path.join(root, 'train_val')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
                file_name = os.path.join(root, 'val')
                self.open_data = datasets.ImageFolder(file_name, transform=self.transform)
            else:  # self.split == 'test'
                file_name = os.path.join(root, 'train_test')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
                file_name = os.path.join(root, 'test')
                self.open_data = datasets.ImageFolder(file_name, transform=self.transform)
        elif self.mode == 'openmany':
            if self.split == 'train':
                file_name = os.path.join(root, 'train_train')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
            elif self.split == 'val':
                file_name = os.path.join(root, 'train_val')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
                file_name = os.path.join(root, 'val')
                self.open_data = datasets.ImageFolder(file_name, transform=self.transform)
            else:  # self.split == 'test'
                file_name = os.path.join(root, 'train_test')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
                file_name = os.path.join(root, 'test')
                self.open_data = datasets.ImageFolder(file_name, transform=self.transform)
        else:
            raise NameError('Unknown mode ({})!'.format(self.mode))
        self.classes = self.data.classes
        self.cls_num = len(self.classes)
        self.closed_samples = len(self.data)
        self.open_classes = []
        self.open_samples = 0
        self.open_cls_num = len(self.open_classes)
        if (self.mode == 'openfew' or self.mode == 'openmany') and (self.split == 'test' or self.split == 'val'):
            self.open_samples = len(self.open_data)
            self.open_classes = self.open_data.classes
            self.open_cls_num = len(self.open_classes)

        # train_val_sample_list = torch.zeros(64).long()
        # if self.split == 'val':
        #     for i in range(len(self.data)):
        #         train_val_sample_list[self.data[i][1]] += 1
        # torch.save(train_val_sample_list, 'train_val_sample_list.pt')
        #
        # train_train_sample_list = 600 * torch.ones(64).long()
        # train_val_sample_list = 300 * torch.ones(64).long()
        # train_test_sample_list = 300 * torch.ones(64).long()
        # val_sample_list = 600 * torch.ones(16).long()
        # test_sample_list = 600 * torch.ones(20).long()

        train_train_sample_list = torch.load('dataset/miniImageNet/train_train_sample_list.pt')
        train_val_sample_list = torch.load('dataset/miniImageNet/train_val_sample_list.pt')
        train_test_sample_list = torch.load('dataset/miniImageNet/train_test_sample_list.pt')
        val_sample_list = torch.load('dataset/miniImageNet/val_sample_list.pt')
        test_sample_list = torch.load('dataset/miniImageNet/test_sample_list.pt')

        if self.split == 'train':
            self.n_sample_list = train_train_sample_list
        elif self.split == 'val':
            if self.mode.startswith('open'):
                self.n_sample_list = torch.cat((train_val_sample_list, val_sample_list), dim=0)
            else:
                self.n_sample_list = train_val_sample_list
        else:  # self.split == 'test'
            if self.mode.startswith('open'):
                self.n_sample_list = torch.cat((train_test_sample_list, test_sample_list), dim=0)
            else:
                self.n_sample_list = train_test_sample_list

    def __getitem__(self, index):
        if self.mode == 'regular':
            return self.data[index]
        elif self.mode.startswith('open'):
            if index < self.closed_samples:
                sample = self.data[index]
                return sample[0], sample[1]
            else:
                sample = self.open_data[index - self.closed_samples]
                return sample[0], sample[1]+self.cls_num

    def __len__(self):
        if (self.mode == 'openfew' or self.mode == 'openmany') and (self.split == 'test' or self.split == 'val'):
            return self.closed_samples + self.open_samples
        else:
            return self.closed_samples
