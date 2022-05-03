import timm
import torch
from torch import nn
from torch.distributions import Beta
import torchaudio as ta

from utils import add_to_catalog

NN_CATALOG = {}


@add_to_catalog('baseline', NN_CATALOG)
class Net(nn.Module):
    def __init__(self, output_len, batch_time_factor,
                 preproc_config=None, backbone_config=None,
                 backbone_path=None):
        super().__init__()
        preproc_config = preproc_config or {}
        backbone_config = backbone_config or {}

        self.audio2image = self._init_audio2image(**preproc_config)
        self.backbone = self._init_backbone(**backbone_config)
        self.load_backbone(backbone_path)
        self.head = self._init_head(self.backbone.feature_info[-1]['num_chs'], output_len)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.mixup = Mixup()
        self.batch_time_factor = batch_time_factor

    def forward(self, wav_tensor, y=None):
        if self.training:
            wav_tensor = self.batch_crop(wav_tensor)

        spectrogram = self.audio2image(wav_tensor)
        spectrogram = spectrogram.permute(0, 2, 1)
        spectrogram = spectrogram[:, None, :, :]

        if self.training:
            spectrogram, y = self.apply_mixup(spectrogram, y)

        x = self.backbone(spectrogram)
        if self.training:
            x = x.permute(0, 2, 1, 3)
            x = self.batch_uncrop(x)
            x = x.permute(0, 2, 1, 3)

        logits = self.head(x)
        loss = None
        if y is not None:
            loss = self.loss(logits, y)

        return {'loss': loss, 'logits': logits.sigmoid()}

    def apply_mixup(self, spectrogram, y):
        spectrogram = spectrogram.permute(0, 2, 1, 3)
        spectrogram = self.batch_uncrop(spectrogram)

        spectrogram, y = self.mixup(spectrogram, y)

        spectrogram = self.batch_crop(spectrogram)
        spectrogram = spectrogram.permute(0, 2, 1, 3)
        return spectrogram, y

    def batch_crop(self, tensor):
        factor = self.batch_time_factor
        b, t = tensor.shape[:2]
        tensor = tensor.reshape(b * factor, t // factor, *tensor.shape[2:])
        return tensor

    def batch_uncrop(self, tensor):
        factor = self.batch_time_factor
        b, t = tensor.shape[:2]
        tensor = tensor.reshape(b // factor, t * factor, *tensor.shape[2:])
        return tensor

    @staticmethod
    def _init_audio2image(sample_rate=32000, n_fft=2048, win_length=2048,
                          hop_length=512, f_min=16, f_max=16386, pad=0,
                          n_mels=256, power=2, normalized=False, top_db=80.):
        mel = ta.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=n_mels,
            power=power,
            normalized=normalized,
        )
        db_scale = ta.transforms.AmplitudeToDB(top_db=top_db)
        audio2image = torch.nn.Sequential(mel, db_scale)
        return audio2image

    @staticmethod
    def _init_backbone(**backbone_kwargs):
        backbone = timm.create_model(
            **backbone_kwargs,
            num_classes=0,
            global_pool="",
            in_chans=1,
        )
        return backbone

    @staticmethod
    def _init_head(input_chs, output_len):
        head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Flatten(),
            torch.nn.Linear(input_chs, output_len)
        )
        return head

    def load_backbone(self, weights_path=None):
        if weights_path:
            state_dict = torch.load(weights_path)
            conv1_weight = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            state_dict.pop('fc.bias')
            state_dict.pop('fc.weight')
            self.backbone.load_state_dict(state_dict)


class Mixup(nn.Module):
    def __init__(self, mix_beta=1):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, sample_weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        mixup_weight = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = mixup_weight.view(-1, 1) * X + (1 - mixup_weight.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = mixup_weight.view(-1, 1, 1) * X + (1 - mixup_weight.view(-1, 1, 1)) * X[perm]
        else:
            X = mixup_weight.view(-1, 1, 1, 1) * X + (1 - mixup_weight.view(-1, 1, 1, 1)) * X[perm]

        Y = mixup_weight.view(-1, 1) * Y + (1 - mixup_weight.view(-1, 1)) * Y[perm]

        if sample_weight is None:
            return X, Y
        else:
            sample_weight = mixup_weight.view(-1) * sample_weight + (1 - mixup_weight.view(-1)) * sample_weight[perm]
            return X, Y, sample_weight
