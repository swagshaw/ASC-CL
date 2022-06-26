import logging
from typing import List

from soundata.datasets import tau2019uas
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
import librosa

logger = logging.getLogger()


def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr)
    return y


class ASC_Dataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame):
        self.data_frame = data_frame
        self.dataset = tau2019uas.Dataset(data_home='data/TAU_ASC')

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        audio_id = self.data_frame.iloc[index]["audio_name"]
        clip = self.dataset.clip(audio_id)

        waveform, sr = clip.audio
        waveform = np.array((waveform[0] + waveform[1]) / 2)
        max_length = sr * 10

        if len(waveform) > max_length:
            waveform = waveform[0:max_length]
        else:
            waveform = np.pad(waveform, (0, max_length - len(waveform)), 'constant')

        tag = clip.tags.labels[0]
        target = self.data_frame.iloc[index]["label"]
        target = np.eye(10)[target]

        data_dict = {
            'audio_name': audio_id, 'waveform': waveform, 'target': target, 'tag': tag}

        return data_dict


class ESC50_Dataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, sr=44100):
        self.data_frame = data_frame
        self.sr = sr

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        audio_name = self.data_frame.iloc[index]["audio_name"]

        waveform = load_audio(audio_name, self.sr)
        tag = self.data_frame.iloc[index]["tag"]
        target = self.data_frame.iloc[index]["label"]
        target = np.eye(50)[target]
        data_dict = {'audio_name': audio_name, 'waveform': waveform, 'target': target, 'tag': tag}

        return data_dict


def default_collate_fn(batch):
    audio_name = [data['audio_name'] for data in batch]
    waveform = [data['waveform'] for data in batch]
    target = [data['target'] for data in batch]

    waveform = torch.FloatTensor(waveform)
    target = torch.FloatTensor(target)

    return {'audio_name': audio_name, 'waveform': waveform, 'target': target}


def get_train_datalist(args, cur_iter: int) -> List:
    datalist = []
    if args.dataset == "ESC-50":
        for test_fold_num in range(1, 6):
            collection_name = get_train_collection_name(
                dataset=args.dataset,
                exp=args.exp_name,
                rnd=args.rnd_seed,
                n_cls=args.n_cls_a_task,
                iter=cur_iter,
            )
            datalist.append(pd.read_json(
                os.path.join(args.data_root, f"{collection_name}_{test_fold_num}.json")
            ).to_dict(orient="records"))
            logger.info(f"[Train] Get datalist from {collection_name}_{test_fold_num}.json")
    else:
        collection_name = get_train_collection_name(
            dataset=args.dataset,
            exp=args.exp_name,
            rnd=args.rnd_seed,
            n_cls=args.n_cls_a_task,
            iter=cur_iter,
        )

        datalist = pd.read_json(os.path.join(args.data_root, f"{collection_name}.json")
                                ).to_dict(orient="records")
        logger.info(f"[Train] Get datalist from {collection_name}.json")

    return datalist


def get_train_collection_name(dataset, exp, rnd, n_cls, iter):
    collection_name = "{dataset}_train_{exp}_rand{rnd}_cls{n_cls}_task{iter}".format(
        dataset=dataset, exp=exp, rnd=rnd, n_cls=n_cls, iter=iter
    )
    return collection_name


def get_test_datalist(args, exp_name: str, cur_iter: int) -> List:
    if exp_name is None:
        exp_name = args.exp_name

    if exp_name == "disjoint":
        # merge current and all previous tasks
        tasks = list(range(cur_iter + 1))
    else:
        raise NotImplementedError

    datalist = []
    if args.dataset == "ESC-50":
        for test_fold_num in range(1, 6):
            fold_list = []
            for iter_ in tasks:
                collection_name = "{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}".format(
                    dataset=args.dataset, rnd=args.rnd_seed, n_cls=args.n_cls_a_task, iter=iter_
                )
                fold_list += pd.read_json(
                    os.path.join(args.data_root, f"{collection_name}_{test_fold_num}.json")
                ).to_dict(orient="records")
                logger.info(f"[Test ] Get datalist from {collection_name}_{test_fold_num}.json")
            datalist.append(fold_list)
    else:
        for iter_ in tasks:
            collection_name = "{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}".format(
                dataset=args.dataset, rnd=args.rnd_seed, n_cls=args.n_cls_a_task, iter=iter_
            )
            datalist += pd.read_json(
                os.path.join(args.data_root, f"{collection_name}.json")
            ).to_dict(orient="records")
            logger.info(f"[Test ] Get datalist from {collection_name}.json")

    return datalist


def get_dataloader(data_frame, dataset, split, batch_size, num_workers=8):
    if dataset == 'TAU-ASC':
        dataset = ASC_Dataset(data_frame=data_frame)
    elif dataset == "ESC-50":
        dataset = ESC50_Dataset(data_frame=data_frame)
    is_train = True if split == 'train' else False

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=is_train, drop_last=False,
                      num_workers=num_workers, collate_fn=default_collate_fn)


if __name__ == '__main__':
    pass
