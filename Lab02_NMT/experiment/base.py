import os
import time
from abc import ABC, abstractmethod
from typing import Type, Dict

import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
import wandb

from dataset.augmentations import (
    Compose,
    OneOf,
    NoiseInjection,
    GaussianNoise,
    PinkNoise,
    RandomVolume,
    Normalize
)
from dataset.bird_clef import BirdDataset
from neural_network.base import save_model, load_model
from utils import Task
from neural_network import NN_CATALOG
from metrics import METRICS_CATALOG


def get_fold(pd_data, fold_i, fold_count, random_state):
    kfold = KFold(fold_count, shuffle=True, random_state=random_state)
    for i, (train_idx, val_idx) in enumerate(kfold.split(pd_data, None)):
        if i == fold_i:
            train_pd = pd_data.iloc[train_idx]
            val_pd = pd_data.iloc[val_idx]
            return train_pd, val_pd


class Experiment(ABC):
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.train_meta = self.read_meta()
        self.all_species = sorted(set(self.train_meta.target_raw.sum()))
        self.task = Task(*self.read_data())
        self.model = self.init_model()
        self.metrics = self.init_metrics()
        self.trainer = self.init_trainer()

        self.stats = dict()
        self.time = None

    def train(self):
        start_time = time.time()
        train_iterator, val_iterator = self.task
        self.trainer.train(train_iterator, val_iterator)
        end_time = time.time()
        self.time = end_time - start_time
        # self.save_model()

    def read_data(self):
        data_config = self.config['data']

        if 'fold' in data_config:
            train_meta, val_meta = get_fold(
                self.train_meta,
                fold_i=data_config['fold'],
                fold_count=5,
                random_state=42,
            )
        elif 'all_data' in data_config:
            _, val_meta = train_test_split(self.train_meta, test_size=0.2, random_state=42)
            train_meta = self.train_meta
        else:
            train_meta, val_meta = train_test_split(self.train_meta, test_size=0.2, random_state=42)

        if 'teacher_path' in data_config:
            teacher_pd = pd.read_csv(data_config['teacher_path'])
            teacher_pd['filt_pred'] = teacher_pd.filt_pred.apply(eval)
        else:
            teacher_pd = None

        train_dataset = BirdDataset(
            train_meta,
            data_root=data_config['wav_root'],
            teacher_pd=teacher_pd,
            crop_len=data_config['crop_len'],
            sample_rate=data_config['sample_rate'],
            augmentations=Compose([
                    OneOf(
                        [
                            NoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=5, max_snr=20),
                            PinkNoise(p=1, min_snr=5, max_snr=20),
                        ],
                        p=0.2,
                    ),
                    RandomVolume(p=0.2, limit=4),
                    Normalize(p=1),
                ]),

        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['n_jobs'],
            pin_memory=False,
            drop_last=True,
        )

        val_dataset = BirdDataset(
            val_meta,
            data_root=data_config['wav_root'],
            crop_len=data_config['crop_len'],
            sample_rate=data_config['sample_rate'],
            augmentations=Compose([Normalize(p=1)])
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['n_jobs'],
            pin_memory=False,
            drop_last=False,
        )

        return train_dataloader, val_dataloader

    def read_meta(self):
        data_config = self.config['data']
        train_meta = pd.read_csv(data_config['meta_path'])
        train_meta.loc[:, 'secondary_labels'] = train_meta.secondary_labels.apply(eval)
        train_meta['target_raw'] = train_meta.secondary_labels + train_meta.primary_label.apply(lambda x: [x])
        all_species = sorted(set(train_meta.target_raw.sum()))
        train_meta['target'] = train_meta.target_raw.apply(lambda species: [int(s in species) for s in all_species])
        return train_meta

    def init_model(self):
        model_config = self.config['model']
        data_config = self.config['data']
        model_class = NN_CATALOG[model_config['name']]

        model = model_class(len(self.all_species), int(data_config['crop_len'] // data_config['test_wav_len']),
                            **model_config['params'])
        if 'model_path' in self.config:
            load_model(model, self.config['model_path'], self.device)
        model.to(self.device)
        return model

    def init_metrics(self):
        score_conf = []
        for metric_name, metric_kwargs in self.config['metric'].items():
            metric_f = METRICS_CATALOG[metric_name]
            score_conf.append((metric_f, metric_kwargs or {}, metric_name))
        return score_conf

    @abstractmethod
    def init_trainer(self):
        raise NotImplementedError

    def save_model(self):
        if wandb.run:
            save_path = os.path.join('model_save', wandb.run.name)
            os.makedirs(save_path, exist_ok=True)
            # torch.save(self.model.state_dict(), os.path.join(save_path, 'final-model.pt'))
            save_model(self.model, os.path.join(save_path, 'final-model.pt'))
        else:
            save_model(self.model, 'final-model.pt')
            # torch.save(self.model.state_dict(), 'final-model.pt')


EXPERIMENT_CATALOG: Dict[str, Type[Experiment]] = {}
