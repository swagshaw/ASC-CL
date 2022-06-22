"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/6/20 下午11:50
@Author  : Yang "Jan" Xiao 
@Description : esc_dataloader
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
import librosa
from tqdm import tqdm


def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr)
    return y


class ESC50_Dataset(Dataset):
    def __init__(self, sr=44100, data_type='train', test_fold_num=1, in_col='filename', out_col='category'):
        dataset_path = 'data/ESC-50-master'
        meta = pd.read_csv(os.path.join(dataset_path, 'meta/esc50.csv'))

        self.data_type = data_type
        if data_type == 'train':
            self.data_list = meta[meta['fold'] != test_fold_num]
        elif data_type == 'test':
            self.data_list = meta[meta['fold'] == test_fold_num]

        print(f'ESC-50 {self.data_type} set using fold {test_fold_num} is creating, using sample rate {sr} Hz ...')

        self.data = []
        self.labels = []
        self.audio_name = []
        self.c2i = {}
        self.i2c = {}
        self.categories = sorted(self.data_list[out_col].unique())

        for i, category in enumerate(self.categories):
            self.c2i[category] = i
            self.i2c[i] = category
        for ind in tqdm(range(len(self.data_list))):
            row = self.data_list.iloc[ind]
            file_path = os.path.join(dataset_path, 'audio', row[in_col])
            self.audio_name.append(row[in_col])
            self.data.append(load_audio(file_path, sr=sr))
            self.labels.append(self.c2i[row['category']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_name = self.audio_name[idx]
        waveform = self.data[idx]
        target = np.eye(50)[self.labels[idx]]
        data_dict = {'audio_name': audio_name, 'waveform': waveform, 'target': target}

        return data_dict


def esc_collate_fn(batch):
    audio_name = [data['audio_name'] for data in batch]
    waveform = [data['waveform'] for data in batch]
    target = [data['target'] for data in batch]

    waveform = torch.FloatTensor(waveform)
    target = torch.FloatTensor(target)

    return {'audio_name': audio_name, 'waveform': waveform, 'target': target}


def get_esc_dataloader(data_type,
                       test_fold_num,
                       batch_size,
                       shuffle=False,
                       drop_last=False,
                       num_workers=8):
    dataset = ESC50_Dataset(data_type=data_type, test_fold_num=test_fold_num, )

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, collate_fn=esc_collate_fn)


if __name__ == '__main__':
    train_loader = get_esc_dataloader(data_type='test', test_fold_num=1, batch_size=32)
    for item in train_loader:
        print(item)
        pass

