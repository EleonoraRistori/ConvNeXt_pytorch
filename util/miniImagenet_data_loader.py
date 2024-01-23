import os.path as osp
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..'))
# IMAGE_PATH = osp.join(ROOT_PATH, './data/miniImagenet/images/')
# SPLIT_PATH = osp.join(ROOT_PATH, './data/miniImagenet/')

IMAGE_PATH = osp.join(ROOT_PATH, './data/mini-imagenet2/')
SPLIT_PATH = osp.join(ROOT_PATH, './data/mini-imagenet2/')


class MiniImageNet(Dataset):
    """ Usage:
    """

    def __init__(self, setname):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        self.transform = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                 np.array([0.229, 0.224, 0.225]))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path))
        return image, label


def load_miniImagenet(batch_size):


    trainset = MiniImageNet('train')
    trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = MiniImageNet('test')
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader, test_loader
