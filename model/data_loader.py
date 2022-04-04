# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        """ Custom Dataset class wrapper for loading the frame characteristics.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, OVP or Youtube.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.datasets = ['../RL-DiVTS/data/OVP/ovp.h5',
                         '../RL-DiVTS/data/Youtube/youtube.h5']
        self.splits_filename = ['../RL-DiVTS/data/splits/' + self.name + '_splits.json']
        self.split_index = split_index

        if 'ovp' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'youtube' in self.splits_filename[0]:
            self.filename = self.datasets[1]

        hdf = h5py.File(self.filename, 'r')
        self.list_features, self.aesthetic_scores_mean = [], []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        for video_name in self.split[self.mode + '_keys']:
            features = torch.Tensor(np.array(hdf[video_name + '/features']))
            aes_scores_mean = torch.Tensor(np.array(hdf[video_name + '/aesthetic_scores_mean']))

            self.list_features.append(features)
            self.aesthetic_scores_mean.append(aes_scores_mean)

        hdf.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.split[self.mode + '_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset, returning:
            the frame features, the video name and the frame-level aesthetic quality scores.

       :param int index: The above-mentioned id of the data.
       """
        video_name = self.split[self.mode + '_keys'][index]
        frame_features = self.list_features[index]
        aesthetic_scores_mean = self.aesthetic_scores_mean[index]
        return frame_features, video_name, aesthetic_scores_mean


def get_loader(mode, video_type, split_index):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, OVP or Youtube.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, video_type, split_index)


if __name__ == '__main__':
    pass
