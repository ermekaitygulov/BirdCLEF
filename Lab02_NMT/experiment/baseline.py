from torch.utils.data import DataLoader

from experiment import EXPERIMENT_CATALOG
from experiment.base import Experiment
from dataset.bird_clef import SpectroDataset
from train_stage import *
from utils import add_to_catalog


@add_to_catalog('baseline', EXPERIMENT_CATALOG)
class Baseline(Experiment):
    def init_trainer(self):
        main_stage = MainStage(self.model, 'main_stage', self.device,
                               self.config['train'], self.metrics)
        # sgd_train_stage = MainStage(self.model, 'sgd_stage', self.device,
        #                             self.config['sgd_train'], self.metrics)
        stage = ComposeStage([main_stage])
        return stage


@add_to_catalog('spec_dataset', EXPERIMENT_CATALOG)
class SpecDataExperiment(Experiment):
    def init_trainer(self):
        spec_stage = SpectroStage(self.model, 'main_stage', self.device,
                                  self.config['train'], self.metrics)
        # sgd_train_stage = MainStage(self.model, 'sgd_stage', self.device,
        #                             self.config['sgd_train'], self.metrics)
        stage = ComposeStage([spec_stage])
        return stage

    def read_data(self):
        train_dataloader, val_dataloader = super(SpecDataExperiment, self).read_data()
        train_dataset = train_dataloader.dataset
        val_dataset = val_dataloader.dataset
        train_dataset = SpectroDataset(
            train_dataset.df,
            train_dataset.data_root,
            train_dataset.crop_len,
            train_dataset.sample_rate,
            train_dataset.augmentations
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_dataloader.batch_size,
            shuffle=True,
            num_workers=train_dataloader.num_workers,
            pin_memory=False,
            drop_last=True,
        )
        val_dataset = SpectroDataset(
            val_dataset.df,
            val_dataset.data_root,
            val_dataset.crop_len,
            val_dataset.sample_rate,
            val_dataset.augmentations
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_dataloader.batch_size,
            shuffle=False,
            num_workers=val_dataloader.num_workers,
            pin_memory=False,
            drop_last=False,
        )
        return train_dataloader, val_dataloader
