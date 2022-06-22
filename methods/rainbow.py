"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/6/22 下午12:33
@Author  : Yang "Jan" Xiao 
@Description : rainbow_memory
"""
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import get_dataloader
from methods.base import BaseMethod


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


logger = logging.getLogger()


class RM(BaseMethod):
    def __init__(self, criterion, device, n_classes, **kwargs):
        super().__init__(criterion, device, n_classes, **kwargs)
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "uncertainty"

    def train(self, n_epoch, batch_size, n_worker):
        if len(self.memory_list) > 0:
            memory_loader = get_dataloader(pd.DataFrame(self.memory_list), self.dataset, split='train',
                                           batch_size=batch_size,
                                           num_workers=n_worker)
            stream_batch_size = batch_size - batch_size // 2
        else:
            memory_loader = None
            stream_batch_size = batch_size

        train_list = self.streamed_list
        test_list = self.test_list

        train_loader, test_loader = self.get_dataloader(
            stream_batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list) + len(self.memory_list)}")
        logger.info(f"Test samples: {len(test_list)}")
        acc_list = []
        best = {'acc': 0, 'epoch': 0}
        for epoch in range(n_epoch):
            mean_loss = 0
            if memory_loader is not None and train_loader is not None:
                data_iterator = zip(train_loader, cycle(memory_loader))
            elif memory_loader is not None:
                data_iterator = memory_loader
            elif train_loader is not None:
                data_iterator = train_loader
            else:
                raise NotImplementedError("None of dataloder is valid")

            for data in tqdm(data_iterator):
                if len(data) == 2:
                    stream_data, mem_data = data
                    x = torch.cat([stream_data["waveform"], mem_data["waveform"]])
                    y = torch.cat([stream_data["target"], mem_data["target"]])
                else:
                    x = data["waveform"]
                    y = data["target"]
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward
                self.model.train()

                batch_output_dict = self.model(x)
                """{'clipwise_output': (batch_size, classes_num), ...}"""
                batch_target_dict = {'target': y}
                """{'target': (batch_size, classes_num)}"""

                # Loss
                loss = self.criterion(batch_output_dict, batch_target_dict)

                # Backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss = loss.item()
                mean_loss += loss

            epoch_loss = mean_loss / len(train_loader)
            logger.info(f'Epoch {epoch} | Training Loss: {epoch_loss}')
            # Evaluate
            test_statistics = self.evaluator.evaluate(test_loader)
            ave_acc = np.mean(test_statistics['accuracy'])
            acc_list.append(ave_acc)
            logger.info(f"Epoch {epoch} | Evaluation Accuracy: {ave_acc}")

            if ave_acc > best['acc']:
                best['acc'] = ave_acc
                best['epoch'] = epoch

            return np.mean(acc_list)
