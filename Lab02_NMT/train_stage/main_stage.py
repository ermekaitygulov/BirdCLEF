from collections import deque
import os

import numpy as np
import torch
import wandb
from torch import optim
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from neural_network.base import save_model


class MainStage:
    default_config = {
        'opt_params': {},
        'log_window_size': 10,
        'opt_class': 'Adam',
        'apex': False,
        'save_path': 'model_save'
    }

    def __init__(self, model, stage_name, device, stage_config, metrics):
        self.config = self.default_config.copy()
        self.config.update(stage_config)
        self.name = stage_name
        self.model = model
        self.opt = self.init_opt()
        self.lr_scheduler = self.init_scheduler()
        self.scaler = GradScaler(enabled=self.config['apex'])
        self.criterion = nn.BCELoss()
        self.device = device
        self.metrics = metrics

    def train(self, train_iterator, val_iterator):
        train_step = 0

        for epoch in range(self.config['epoch']):
            train_step = self.train_epoch(
                train_iterator,
                train_step
            )
            with torch.no_grad():
                self.val_epoch(
                    val_iterator,
                    epoch,
                )

            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_model(f'{self.name}-{epoch+1}-model.pt')

            print(f'Epoch: {epoch + 1:02}')
        with torch.no_grad():
            print(self.val_epoch(val_iterator, None, log_wandb=False))
        self.save_model('final-model.pt')

    def train_epoch(self, iterator, global_step):
        self.model.train()

        tqdm_iterator = tqdm(iterator)
        loss_window = deque(maxlen=self.config['log_window_size'])
        for i, batch in enumerate(tqdm_iterator):
            self.opt.zero_grad()
            with autocast(enabled=self.config['apex']):
                loss = self.compute_batch_loss(batch)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            loss_window.append(loss.item())
            if (i + 1) % self.config['log_window_size'] == 0:
                log_dict = dict()
                log_dict[f'train_loss'] = np.mean(loss_window)
                log_dict['train_step'] = global_step
                log_dict['learning_rate'] = self.opt.param_groups[0]["lr"]

                if wandb.run:
                    wandb.log({self.name: log_dict})
                tqdm_iterator.set_postfix(train_loss=log_dict['train_loss'])

            global_step += 1

        return global_step

    def val_epoch(self, dataloader, epoch, log_wandb=True):
        tqdm_dataloader = tqdm(dataloader)
        self.model.eval()
        y_true = None
        y_pred = None

        for batch in tqdm_dataloader:
            logits = self.model(batch['wav'].to(self.device))['logits']
            batch_target = batch['target'].cpu().numpy()
            batch_pred = logits.cpu().numpy()

            if y_true is None:
                y_true = batch_target
                y_pred = batch_pred
            else:
                y_true = np.vstack((y_true, batch_target))
                y_pred = np.vstack((y_pred, batch_pred))

        val_metrics = self.score_pred(y_true, y_pred)
        np.save('last_pred.npy', y_pred)
        np.save('last_true.npy', y_true)
        if wandb.run and log_wandb:
            wandb.log({self.name: {**val_metrics, 'epoch': epoch}})
        return val_metrics

    def score_pred(self, y_true, y_pred):
        score_dict = {}
        for score_f, score_kwargs, score_prefix in self.metrics:
            score_dict.update({
                f'{score_prefix}-{t}': score_f(y_true, y_pred > t, **score_kwargs)
                for t in self.config['score_trsh']
            })
        return score_dict

    def compute_batch_loss(self, batch):
        loss = self.model(
            batch['wav'].to(self.device),
            batch['target'].to(self.device)
        )['loss']
        return loss

    def init_opt(self):
        opt_class = getattr(optim, self.config['opt_class'])
        opt_params = self.config['opt_params']
        opt = opt_class(self.model.parameters(), **opt_params)
        return opt

    def init_scheduler(self):
        # TODO refactor
        if 'scheduler_class' not in self.config:
            return None
        scheduler_class = getattr(optim.lr_scheduler, self.config['scheduler_class'])
        scheduler_params = self.config['scheduler_params']
        scheduler = scheduler_class(self.opt, **scheduler_params)
        return scheduler

    def save_model(self, save_name):
        save_path = self.config['save_path']
        if wandb.run:
            save_path = os.path.join(self.config['save_path'], wandb.run.name)
        os.makedirs(save_path, exist_ok=True)
        save_model(self.model, os.path.join(save_path, save_name))