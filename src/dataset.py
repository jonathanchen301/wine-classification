from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch


class wineDataset:

    def __init__(self, root, headers=None, transform=None):
        dataset = pd.read_csv(root, names=headers)
        self.nsamples = dataset.shape[0]
        self.transform = transform
        self.X = dataset.iloc[:, 1:]
        self.y = dataset.iloc[:, 0]

    def __getitem__(self, index):
        sample = (self.X.iloc[index, :], self.y.iloc[index])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.nsamples


class OneHotEncoding:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        features, label = sample
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.zeros(self.num_classes)
        label[int(sample[1]) - 1] = 1
        return features, label
