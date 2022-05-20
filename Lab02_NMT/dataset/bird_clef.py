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


class SpectroDataset(BirdDataset):
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

        # wav = torch.tensor(wav)
        target = torch.tensor(target, dtype=torch.float)
        return {
            **self.audio2image(wav),
            'target': target,
        }

    def audio2image(self, wav):
        melspec = librosa.feature.melspectrogram(
            y=wav,
            sr=32000,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window="hann",
            center=False,
            pad_mode="constant",
            power=2.0,
            n_mels=256,
            fmin=16.,
            fmax=16386.,
        )
        pcen = librosa.pcen(
            melspec,
            sr=32000
        )
        clean_mel = librosa.power_to_db(melspec ** 1.5)
        melspec = librosa.power_to_db(melspec)

        # norm_melspec = normalize_melspec(melspec)
        # norm_pcen = normalize_melspec(pcen)
        # norm_clean_mel = normalize_melspec(clean_mel)
        image = np.stack([melspec, pcen, clean_mel], axis=0)
        return {'melspec': torch.tensor(melspec), 'melspec_ext': torch.tensor(image).float()}

# def normalize_melspec(X: np.ndarray):
#     eps = 1e-6
#     mean = X.mean()
#     X = X - mean
#     std = X.std()
#     Xstd = X / (std + eps)
#     norm_min, norm_max = Xstd.min(), Xstd.max()
#     if (norm_max - norm_min) > eps:
#         V = Xstd
#         V[V < norm_min] = norm_min
#         V[V > norm_max] = norm_max
#         V = 255 * (V - norm_min) / (norm_max - norm_min)
#         V = V.astype(np.uint8)
#     else:
#         # Just zero
#         V = np.zeros_like(Xstd, dtype=np.uint8)
#     return V