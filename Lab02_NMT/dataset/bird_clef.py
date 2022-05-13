import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


def load_wav(fpath, offset, duration):
    wav, sr = librosa.load(fpath, sr=None, offset=offset, duration=duration)
    assert sr <= 32000, sr
    return wav, sr


class BirdDataset(Dataset):
    def __init__(self, df, data_root, crop_len=30, sample_rate=32000, augmentations=None):
        super().__init__()
        self.df = df
        self.data_root = data_root
        self.crop_len = crop_len
        self.sample_rate = sample_rate
        self.augmentations = augmentations

    def __getitem__(self, idx):
        fname = self.df.iloc[idx]['filename']
        fpath = os.path.join(self.data_root, fname)
        wav_len = self.df.iloc[idx]['duration']

        max_offset = max(0, wav_len - self.crop_len)
        random_offset = random.randint(0, max_offset)

        wav, sr = load_wav(fpath, random_offset, self.crop_len)
        if self.augmentations:
            try:
                wav = self.augmentations(wav, None)
            except ValueError as e:
                print(random_offset)
                raise e

        to_pad = self.crop_len * self.sample_rate - wav.shape[0]
        if to_pad > 0:
            wav = np.pad(wav, (0, to_pad))

        target = self.df.iloc[idx]['target']

        # TODO: add weighting

        wav = torch.tensor(wav)
        target = torch.tensor(target, dtype=torch.float)
        return {
            'wav': wav,
            'target': target,
        }

    def __len__(self):
        return len(self.df)
