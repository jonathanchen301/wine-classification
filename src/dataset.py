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


class toTensor:

    def __call__(self, sample):
        return torch.tensor(sample[0], dtype=torch.float32), torch.tensor(
            sample[1], dtype=torch.float32
        )
