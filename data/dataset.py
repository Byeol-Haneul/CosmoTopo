'''
Author: Jun-Young Lee

CustomDataset for loading combinatorial complexes. 
Supports augmentation using the neighborhood dropping technique in utils.augmentation.
'''

import torch
from torch.utils.data import Dataset
from utils.augmentation import augment_data
import h5py

class CustomDataset(Dataset):
    def __init__(self, data, feature_names):
        self.feature_names = feature_names
        self.dataset = [
            {name: sample[i] for i, name in enumerate(self.feature_names)}
            for sample in data
        ]
        self.augmented_dataset = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.augmented_dataset[idx]

    def augment(self, drop_prob=0.1, cci_mode='euclidean'):
        self.augmented_dataset = [
            augment_data(sample, drop_prob, cci_mode) for sample in self.dataset
        ]
