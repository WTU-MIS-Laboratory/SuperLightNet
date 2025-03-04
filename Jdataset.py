
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import os
import torch
import random
import numpy as np
from torchvision.transforms import transforms


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (3, 2), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (3, 2)), mode='constant')
        return {'image': image, 'label': label}


class Center_Crop(object):
    output_size = (128, 128, 128)
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (w, h, d, c) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2],:]

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']
        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


def BraTS_processing_train(sample):
    trans = transforms.Compose([
        Pad(),
        Center_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        ToTensor()
    ])
    sample = trans(sample)
    return sample


def BraTS_processing_valid(sample):
    trans = transforms.Compose([
        Pad(),
        Center_Crop(),
        ToTensor()
    ])
    sample = trans(sample)
    return sample


class BraTS(Dataset):
    def __init__(self, list_file, root='',mode=''):
        self.lines = []
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                paths.append(path)
                self.lines.append(line)
        self.names = names
        self.paths = paths
        self.mode = mode

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path + 'pkl_ui8f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = BraTS_processing_train(sample)
            return sample['image'], sample['label']
        elif self.mode == 'valid':
            image, label = pkload(path + 'pkl_ui8f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = BraTS_processing_valid(sample)
            return sample['image'], sample['label']
        else:
            raise ValueError("mode not set.")

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
