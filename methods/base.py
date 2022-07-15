"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/6/19 下午6:48
@Author  : Yang "Jan" Xiao 
@Description : base
When we make a new one, we should inherit the BaseMethod class.
"""
import logging
import random
import numpy as np
import pandas as pd
import torch
from soundata.datasets import tau2019uas
from torch import optim
from tqdm import tqdm

from torch_audiomentations import Compose, PitchShift, Shift, AddColoredNoise
from audiomentations import Compose as Normal_Compose
from audiomentations import FrequencyMask
from data_loader import get_dataloader, load_audio
from utils.evaluate import Evaluator

logger = logging.getLogger()


class BaseMethod:
    def __init__(
            self, criterion, device, n_classes, model, **kwargs
    ):
        # Parameters for Dataloader
        self.num_learned_class = 0
        self.num_learning_class = kwargs["n_init_cls"]
        self.learned_classes = []
        self.class_mean = [None] * n_classes
        self.exposed_classes = []
        self.seen = 0
        self.dataset = kwargs["dataset"]

        # Parameters for Trainer
        self.patience = 7
        self.device = device
        self.criterion = criterion
        self.lr = kwargs["lr"]
        self.optimizer, self.scheduler = None, None
        # self.criterion = self.criterion.to(self.device)
        self.evaluator = Evaluator(model=model)
        self.counter = 0

        # Parameters for Model
        self.model_name = kwargs["model_name"]
        self.model = model
        self.model = self.model.to(self.device)

        # Parameters for Prototype Sampler
        self.feature_extractor = model
        self.sample_length = 48000
        if self.dataset == 'ESC-50':
            self.sample_length = 44100

        # Parameters for Memory Updating
        self.prev_streamed_list = []
        self.streamed_list = []
        self.test_list = []
        self.memory_list = []
        self.memory_size = kwargs["memory_size"]
        self.mem_manage = kwargs["mem_manage"]
        self.already_mem_update = False
        self.mode = kwargs["mode"]
        if self.mode == "finetune":
            self.memory_size = 0
            self.mem_manage = "random"
        self.uncert_metric = kwargs["uncert_metric"]
        self.metric_k = kwargs["metric_k"]
        self.noise_lambda = kwargs["noise_lambda"]

    def set_current_dataset(self, train_datalist, test_datalist):
        random.shuffle(train_datalist)
        self.prev_streamed_list = self.streamed_list
        self.streamed_list = train_datalist
        self.test_list = test_datalist

    def before_task(self, datalist, init_opt=True):
        logger.info("Apply before_task")

        # Confirm incoming classes
        incoming_classes = pd.DataFrame(datalist)["tag"].unique().tolist()
        self.exposed_classes = list(set(self.learned_classes + incoming_classes))
        self.num_learning_class = max(
            len(self.exposed_classes), self.num_learning_class
        )

        self.model.num_class = self.num_learning_class
        self.model = self.model.to(self.device)
        if init_opt:
            # reinitialize the optimizer and scheduler
            logger.info("Reset the optimizer and scheduler states")
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

        logger.info(f"Increasing the head of fc {self.learned_classes} -> {self.num_learning_class}")

        self.already_mem_update = False

    def after_task(self, cur_iter):
        logger.info("Apply after_task")
        self.learned_classes = self.exposed_classes
        self.num_learned_class = self.num_learning_class
        self.update_memory(cur_iter)

    def update_memory(self, cur_iter, num_class=None):
        if num_class is None:
            num_class = self.num_learning_class

        if not self.already_mem_update:
            logger.info(f"Update memory over {num_class} classes by {self.mem_manage}")
            candidates = self.streamed_list + self.memory_list
            if len(candidates) <= self.memory_size:
                self.memory_list = candidates
                self.seen = len(candidates)
                logger.warning("Candidates < Memory size")
            else:
                if self.mem_manage == "random":
                    self.memory_list = self.rnd_sampling(candidates)
                elif self.mem_manage == "reservoir":
                    self.reservoir_sampling(self.streamed_list)
                elif self.mem_manage == "prototype":
                    self.memory_list = self.mean_feature_sampling(
                        exemplars=self.memory_list,
                        samples=self.streamed_list,
                        num_class=num_class,
                    )
                elif self.mem_manage == "uncertainty":
                    if cur_iter == 0:
                        self.memory_list = self.equal_class_sampling(
                            candidates, num_class
                        )
                    else:
                        self.memory_list = self.uncertainty_sampling(
                            candidates,
                            num_class=num_class,
                        )
                else:
                    logger.error("Not implemented memory management")
                    raise NotImplementedError

            assert len(self.memory_list) <= self.memory_size
            logger.info("Memory statistic")
            memory_df = pd.DataFrame(self.memory_list)
            if len(self.memory_list) > 0:
                logger.info(f"\n{memory_df.tag.value_counts(sort=True)}")
            # memory update happens only once per task iteratin.
            self.already_mem_update = True
        else:
            logger.warning(f"Already updated the memory during this iter ({cur_iter})")

    def get_dataloader(self, batch_size, n_worker, train_list, test_list):
        train_loader = get_dataloader(pd.DataFrame(train_list), self.dataset, split='train', batch_size=batch_size,
                                      num_workers=n_worker)
        test_loader = get_dataloader(pd.DataFrame(test_list), self.dataset, split='test', batch_size=128,
                                     num_workers=n_worker)
        return train_loader, test_loader

    def train(self, n_epoch, batch_size, n_worker):
        self.counter = 0
        train_list = self.streamed_list + self.memory_list
        random.shuffle(train_list)
        test_list = self.test_list
        train_loader, test_loader = self.get_dataloader(
            batch_size, n_worker, train_list, test_list
        )
        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)}")
        logger.info(f"Test samples: {len(test_list)}")
        acc_list = []
        best = {'acc': 0, 'epoch': 0}
        for epoch in range(n_epoch):
            mean_loss = 0
            for batch_data_dict in tqdm(train_loader):
                batch_data_dict['waveform'] = batch_data_dict['waveform'].to(self.device)
                batch_data_dict['target'] = batch_data_dict['target'].to(self.device)

                # Forward
                self.model.train()

                batch_output_dict = self.model(batch_data_dict['waveform'], training=True)
                """{'clipwise_output': (batch_size, classes_num), ...}"""
                batch_target_dict = {'target': batch_data_dict['target']}
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
                logger.info(f'Best Accuracy: {ave_acc} in epoch {epoch}.')
                self.counter = 0
            else:
                self.counter += 1
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}.')
                if self.counter >= self.patience:
                    break
        return best['acc']

    def rnd_sampling(self, samples):
        random.shuffle(samples)
        return samples[: self.memory_size]

    def reservoir_sampling(self, samples):
        for sample in samples:
            if len(self.memory_list) < self.memory_size:
                self.memory_list += [sample]
            else:
                j = np.random.randint(0, self.seen)
                if j < self.memory_size:
                    self.memory_list[j] = sample
            self.seen += 1

    def mean_feature_sampling(self, exemplars, samples, num_class):
        """Prototype sampling

        Args:
            features ([Tensor]): [features corresponding to the samples]
            samples ([Datalist]): [datalist for a class]

        Returns:
            [type]: [Sampled datalist]
        """

        def _reduce_exemplar_sets(exemplars, mem_per_cls):
            if len(exemplars) == 0:
                return exemplars

            exemplar_df = pd.DataFrame(exemplars)
            ret = []
            for y in range(self.num_learned_class):
                cls_df = exemplar_df[exemplar_df["label"] == y]
                ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                    orient="records"
                )

            num_dups = pd.DataFrame(ret).duplicated().sum()
            if num_dups > 0:
                logger.warning(f"Duplicated samples in memory: {num_dups}")

            return ret

        mem_per_cls = self.memory_size // num_class
        exemplars = _reduce_exemplar_sets(exemplars, mem_per_cls)
        old_exemplar_df = pd.DataFrame(exemplars)

        new_exemplar_set = []
        sample_df = pd.DataFrame(samples)
        for y in range(self.num_learning_class):
            cls_samples = []
            cls_exemplars = []
            if len(sample_df) != 0:
                cls_samples = sample_df[sample_df["label"] == y].to_dict(
                    orient="records"
                )
            if len(old_exemplar_df) != 0:
                cls_exemplars = old_exemplar_df[old_exemplar_df["label"] == y].to_dict(
                    orient="records"
                )

            if len(cls_exemplars) >= mem_per_cls:
                new_exemplar_set += cls_exemplars
                continue

            # Assign old exemplars to the samples
            cls_samples += cls_exemplars
            if len(cls_samples) <= mem_per_cls:
                new_exemplar_set += cls_samples
                continue

            features = []
            self.feature_extractor.eval()
            with torch.no_grad():
                for data in cls_samples:
                    if self.dataset == 'TAU-ASC':
                        tau_dataset = tau2019uas.Dataset(data_home='data/TAU_ASC')
                        clip = tau_dataset.clip(data['audio_name'])
                        waveform, _ = clip.audio
                        waveform = np.array((waveform[0] + waveform[1]) / 2)
                        max_length = self.sample_length * 10
                        if len(waveform) > max_length:
                            waveform = waveform[0:max_length]
                        else:
                            waveform = np.pad(waveform, (0, max_length - len(waveform)), 'constant')
                    else:
                        waveform = load_audio(data['audio_name'], 44100)
                    waveform = torch.as_tensor(waveform, dtype=torch.float32)
                    waveform = waveform.to(self.device)
                    feature = (
                        self.feature_extractor(waveform.unsqueeze(0))['embedding'].detach().cpu().numpy()
                    )
                    feature = feature / np.linalg.norm(feature, axis=1)  # Normalize
                    features.append(feature.squeeze())

            features = np.array(features)
            logger.debug(f"[Prototype] features: {features.shape}")

            # do not replace the existing class mean
            if self.class_mean[y] is None:
                cls_mean = np.mean(features, axis=0)
                cls_mean /= np.linalg.norm(cls_mean)
                self.class_mean[y] = cls_mean
            else:
                cls_mean = self.class_mean[y]
            assert cls_mean.ndim == 1

            phi = features
            mu = cls_mean
            # select exemplars from the scratch
            exemplar_features = []
            num_exemplars = min(mem_per_cls, len(cls_samples))
            for j in range(num_exemplars):
                S = np.sum(exemplar_features, axis=0)
                mu_p = 1.0 / (j + 1) * (phi + S)
                mu_p = mu_p / np.linalg.norm(mu_p, axis=1, keepdims=True)

                dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
                i = np.argmin(dist)

                new_exemplar_set.append(cls_samples[i])
                exemplar_features.append(phi[i])

                # Avoid to sample the duplicated one.
                del cls_samples[i]
                phi = np.delete(phi, i, 0)

        return new_exemplar_set

    def uncertainty_sampling(self, samples, num_class):
        """uncertainty based sampling

        Args:
            samples ([list]): [training_list + memory_list]
        """
        self.montecarlo(samples, uncert_metric=self.uncert_metric)

        sample_df = pd.DataFrame(samples)
        mem_per_cls = self.memory_size // num_class  # kc: the number of the samples of each class

        ret = []
        """
        Sampling class by class
        """
        for i in range(num_class):
            cls_df = sample_df[sample_df["label"] == i]
            if len(cls_df) <= mem_per_cls:
                ret += cls_df.to_dict(orient="records")
            else:
                jump_idx = len(cls_df) // mem_per_cls
                uncertain_samples = cls_df.sort_values(by="uncertainty")[::jump_idx]
                ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.audio_name.isin(pd.DataFrame(ret).audio_name)]
                .sample(n=num_rest_slots)
                .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).audio_name.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret

    def _compute_uncert(self, infer_list, infer_transform, uncert_name):
        batch_size = 128
        infer_df = pd.DataFrame(infer_list)
        infer_loader = get_dataloader(infer_df, self.dataset, split='test', batch_size=batch_size)

        self.model.eval()
        with torch.no_grad():
            for n_batch, batch_data_dict in enumerate(infer_loader):
                if self.uncert_metric != "noisytune":
                    batch_data_dict['waveform'] = infer_transform(batch_data_dict['waveform'].unsqueeze(1),
                                                                  self.sample_length)
                    batch_data_dict['waveform'] = torch.as_tensor(batch_data_dict['waveform'], dtype=torch.float32)
                    batch_data_dict['waveform'] = batch_data_dict['waveform'].squeeze()
                    batch_data_dict['waveform'] = batch_data_dict['waveform'].to(self.device)
                    logit = self.model(batch_data_dict['waveform'])
                    logit = logit['clipwise_output'].detach().cpu()
                    """{'clipwise_output': (batch_size, classes_num), ...}"""
                    for i, cert_value in enumerate(logit):
                        sample = infer_list[batch_size * n_batch + i]
                        sample[uncert_name] = 1 - cert_value
                else:
                    batch_data_dict['waveform'] = batch_data_dict['waveform'].to(self.device)
                    logit = self.model(input=batch_data_dict['waveform'], training=False, add_noise=True,
                                       noise_lambda=self.noise_lambda,
                                       k=self.metric_k)
                    logit = logit['clipwise_output']
                    for j in range(len(logit)):
                        logit[j] = logit[j].detach().cpu()
                        uncert_name = f"uncert_{str(j)}"
                        for i, cert_value in enumerate(logit[j]):
                            sample = infer_list[batch_size * n_batch + i]
                            sample[uncert_name] = 1 - cert_value

    def montecarlo(self, candidates, uncert_metric="shift"):
        transform_cands = []
        logger.info(f"Compute uncertainty by {uncert_metric}!")
        if uncert_metric == "shift":
            transform_cands = [PitchShift(sample_rate=self.sample_length, p=1.0),
                               Shift(sample_rate=self.sample_length, p=1.0)
                               ] * (self.metric_k // 2)
            for idx, tr in enumerate(transform_cands):
                _tr = Compose([tr])
                self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")
        elif uncert_metric == "noise":
            transform_cands = [AddColoredNoise(sample_rate=self.sample_length, p=1.0)] * self.metric_k
            for idx, tr in enumerate(transform_cands):
                _tr = Compose([tr])
                self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")
        elif uncert_metric == "mask":
            transform_cands = [TimeMask(min_band_part=0, max_band_part=0.1),
                               FrequencyMask(min_frequency_band=0, max_frequency_band=0.1, p=1)] * (self.metric_k // 2)
            for idx, tr in enumerate(transform_cands):
                _tr = Normal_Compose([tr])
                self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")
        elif uncert_metric == "combination":
            transform_cands = [TimeMask(min_band_part=0, max_band_part=0.1),
                               FrequencyMask(min_frequency_band=0, max_frequency_band=0.1, p=1),
                               AddColoredNoise(sample_rate=self.sample_length, p=1.0),
                               AddColoredNoise(sample_rate=self.sample_length, p=1.0),
                               PitchShift(sample_rate=self.sample_length, p=1.0),
                               Shift(sample_rate=self.sample_length, p=1.0)
                               ]
            random.shuffle(transform_cands)
            transform_cands = transform_cands[:self.metric_k]
            for idx, tr in enumerate(transform_cands):
                if 'audiomentations' in str(tr):
                    _tr = Normal_Compose([tr])
                else:
                    _tr = Compose([tr])
                self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")
        elif uncert_metric == "noisytune":
            self._compute_uncert(candidates, None, uncert_name=None)

        n_transforms = self.metric_k

        for sample in candidates:
            self.variance_ratio(sample, n_transforms)

    def variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()

    def equal_class_sampling(self, samples, num_class):
        mem_per_cls = self.memory_size // num_class
        sample_df = pd.DataFrame(samples)
        # Warning: assuming the classes were ordered following task number.
        ret = []
        for y in range(self.num_learning_class):
            cls_df = sample_df[sample_df["label"] == y]
            ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                orient="records"
            )

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.audio_name.isin(pd.DataFrame(ret).audio_name)]
                .sample(n=num_rest_slots)
                .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).audio_name.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret


class TimeMask:
    def __init__(self, min_band_part=0.0, max_band_part=0.5):
        self.min_band_part = min_band_part
        self.max_band_part = max_band_part

    def __call__(self, samples, sample_rate):
        num_samples = samples.shape[-1]
        t = random.randint(
            int(num_samples * self.min_band_part),
            int(num_samples * self.max_band_part),
        )
        t0 = random.randint(
            0, num_samples - t
        )
        new_samples = samples.clone()
        mask = torch.zeros(t)
        new_samples[..., t0: t0 + t] *= mask
        return new_samples
