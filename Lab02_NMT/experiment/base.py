import os
import random
import time
from abc import ABC, abstractmethod
from typing import Type, Dict
import yaml

import torch
import wandb
from nltk import WordPunctTokenizer
from nltk.translate.bleu_score import corpus_bleu
from torchtext.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm
import numpy as np

from utils import Task, get_text
from neural_network import NN_CATALOG


class Experiment(ABC):
    def __init__(self, config, device):
        self.config = config
        self.tokenizer = WordPunctTokenizer()
        self.device = device

        self.source = self.init_field()
        self.target = self.init_field()
        self.task = Task(*self.read_data())
        self.model = self.init_model()
        self.trainer = self.init_trainer()
        self.stats = dict()
        self.time = None

    def train(self):
        start_time = time.time()
        train_iterator, val_iterator, _ = self.task
        self.trainer.train(train_iterator, val_iterator)
        end_time = time.time()
        self.time = end_time - start_time
        self.save_model()

    def test(self):
        _, _, test_iterator = self.task

        original_text = []
        generated_text = []
        inference_speed = []
        self.model.eval()
        tqdm_iterator = tqdm(enumerate(test_iterator))

        with torch.no_grad():
            for i, batch in tqdm_iterator:
                src = batch.src
                trg = batch.trg

                _, output = self.model.gen_translate(src, trg, greedy=True)

                # trg = [trg sent len, batch size]
                # output = [trg sent len, batch size, output dim]
                original_text.extend([self.decode_translation(x) for x in trg.cpu().numpy().T])
                generated_text.extend([self.decode_translation(x) for x in output.cpu().numpy().T])
                if tqdm_iterator._ema_dt():
                    inference_speed.append(tqdm_iterator._ema_dn() / tqdm_iterator._ema_dt())

        bleu = corpus_bleu([[text] for text in original_text], generated_text) * 100
        print(f'Bleu: {bleu:.3f}')
        self.stats['bleu'] = bleu
        self.stats['time_spent (min)'] = self.time // 60
        self.stats['time_spent (sec)'] = self.time
        self.stats['inference_speed (it/seq)'] = float(np.mean(inference_speed))
        self.wandb_log_stats()
        self.save_stats()
        self.save_config(f'{int(bleu)}_bleu.yaml')

    def read_data(self):
        data_config = self.config['data']

        dataset = TabularDataset(
            path=data_config['path'],
            format='tsv',
            fields=[('trg', self.target), ('src', self.source)]
        )
        split_ratio = [data_config[f'{split}_size'] for split in ['train', 'val', 'test']]

        random.seed(42)
        random_state = random.getstate()
        train_data, valid_data, test_data = dataset.split(split_ratio=split_ratio, random_state=random_state)
        self.source.build_vocab(train_data, min_freq=data_config['word_min_freq'])
        self.target.build_vocab(train_data, min_freq=data_config['word_min_freq'])
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=data_config['batch_size'],
            device=self.device,
            sort_key=self._len_sort_key,
        )
        return train_iterator, valid_iterator, test_iterator

    def init_model(self):
        model_config = self.config['model']
        model_class = NN_CATALOG[model_config['name']]

        input_dim = len(self.source.vocab)
        output_dim = len(self.target.vocab)
        model = model_class(input_dim, output_dim, self.device, self.target.vocab, **model_config['params'])
        if 'model_path' in self.config:
            model.load(self.config['model_path'])
        model.to(self.device)
        return model

    @abstractmethod
    def init_trainer(self):
        raise NotImplementedError

    def init_field(self):
        field = Field(tokenize=self.tokenize,
                      init_token='<sos>',
                      eos_token='<eos>',
                      lower=True)
        return field

    def wandb_log_stats(self):
        if not wandb.run:
            return
        data = []
        columns = []
        for key, value in self.stats.items():
            data.append(value)
            columns.append(key)
        table = wandb.Table(
            data=[data],
            columns=columns
        )
        wandb.log({"bleu_score": wandb.plot.scatter(
            table,
            "time_spent (min)",
            "bleu", title="BLEU score")
        })

    def save_stats(self):
        save_path = './'
        if wandb.run:
            save_path = os.path.join('model_save', wandb.run.name)
            os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'stats.yaml'), 'w') as fout:
            yaml.dump(self.stats, fout)

    def save_config(self, config_name):
        save_path = './'
        if wandb.run:
            save_path = os.path.join('model_save', wandb.run.name)
            os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, config_name), 'w') as fout:
            yaml.dump(self.config, fout)

    def save_model(self):
        if wandb.run:
            save_path = os.path.join('model_save', wandb.run.name)
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(save_path, 'final-model.pt'))
        else:
            torch.save(self.model.state_dict(), 'final-model.pt')

    def tokenize(self, x):
        token_collection = self.tokenizer.tokenize(x.lower())
        return token_collection

    def decode_translation(self, idx_text):
        vocab = self.target.vocab
        return get_text(idx_text, vocab)

    @staticmethod
    def _len_sort_key(x):
        return len(x.src)


EXPERIMENT_CATALOG: Dict[str, Type[Experiment]] = {}
