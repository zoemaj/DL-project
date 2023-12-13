
from abc import ABC
from PIL import Image
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.tumors.utils  import *
from datasets.dataset import *

class BHDataset(FewShotDataset, ABC):
    _dataset_name = 'break_his'
    _dataset_url = 'https://drive.google.com/file/d/1FlHhtTXzKgQCjxn18j1b4T3Q5FhylIn4/view'

    def load_break_his(self, mode='train', magnification=40, min_samples=20):

        path_embed = f'data/break_his/embeddings/{mode}_data_{magnification}X.pt'
        path_labels = f'data/break_his/embeddings/{mode}_labels_{magnification}X.pt'
        file_name= 'data/break_his'

        # Check whether embeddings already exist
        if os.path.exists(path_embed) and os.path.exists(path_labels):
            print(f"Embeddings for {mode} already exist. Loading from saved files.")
            data_tensor = torch.load(path_embed)
            labels_tensor = torch.load(path_labels)
            return data_tensor, labels_tensor

        # Embeddings don't exist, create them
        else:
            print("We have to create the embeddings first.")
            data_tensor, labels_tensor = load_images(mode,  file_name, magnification)
            return data_tensor, labels_tensor

class BHSimpleDataset(BHDataset):
    def __init__(self, batch_size: int, root: str='./data/', mode: str='train', min_samples: int=20):
        """
        Initialize the dataset with the specified batch size, root directory, mode, and minimum number of samples.
        """
        self.magnification = 400 
        self.initialize_data_dir(root, download_flag=False)
        self.samples, self.targets = self.load_break_his(mode, self.magnification, min_samples)
        self.batch_size = batch_size
        super().__init__()

    def __getitem__(self, i: int):
        """
        Get the i-th sample from the dataset.
        """
        return self.samples[i], self.targets[i]

    def __len__(self):
        """
        Get the number of samples in the dataset.
        """
        return self.samples.shape[0]

    @property
    def dim(self):
        """
        Get the dimension of the dataset.
        """
        return self.samples.shape[1]

    def get_data_loader(self):
        """
        Get a DataLoader for the dataset.
        """
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)

        return data_loader

class BHSetDataset(BHDataset):
    def __init__(self, n_way: int, n_support: int, n_query: int, n_episode: int=100, root: str='./data', mode: str='train'):
        """
        Initialize the dataset with the specified number of ways, support, query, episodes, root directory, and mode.
        """
        self.initialize_data_dir(root, download_flag=True)
        self.magnification = 400
        self.n_way = n_way
        self.n_episode = n_episode
        min_samples = n_support + n_query

        samples_all, targets_all = self.load_break_his(mode, self.magnification, min_samples)
        self.categories = np.unique(targets_all)  # Unique cell labels
        self.x_dim = samples_all.shape[1]

        self.sub_dataloader = []

        sub_data_loader_params = dict(batch_size=min_samples,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.categories:
            samples = samples_all[targets_all == cl, ...]
            sub_dataset = FewShotSubDataset(samples, cl)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

        super().__init__()

    def __getitem__(self, i: int):
        """
        Get the i-th sample from the dataset.
        """
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        """
        Get the number of samples in the dataset.
        """
        return len(self.categories)

    @property
    def dim(self):
        """
        Get the dimension of the dataset.
        """
        return self.x_dim

    def get_data_loader(self):
        """
        Get a DataLoader for the dataset.
        """
        sampler = EpisodicBatchSampler(len(self), self.n_way, self.n_episode)   
        data_loader_params = dict(batch_sampler=sampler, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)
        return data_loader
