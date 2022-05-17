import timm
import torch
from torch import nn
from torch.distributions import Beta
import torchaudio as ta

from utils import add_to_catalog

NN_CATALOG = {}


def save_model(model, path):
    # torch.jit.script(model).save(path)
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)


@add_to_catalog('baseline', NN_CATALOG)
class Net(nn.Module):
    def __init__(self, output_len, batch_time_factor,
                 preproc_config=None, backbone_config=None,
                 backbone_path=None, batch_time_crop=True):
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
        self.batch_time_crop = batch_time_crop

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
        if not self.batch_time_crop:
            return tensor

        factor = self.batch_time_factor
        b, t = tensor.shape[:2]
        tensor = tensor.reshape(b * factor, t // factor, *tensor.shape[2:])
        return tensor

    def batch_uncrop(self, tensor):
        if not self.batch_time_crop:
            return tensor

        factor = self.batch_time_factor
        b, t = tensor.shape[:2]
        tensor = tensor.reshape(b // factor, t * factor, *tensor.shape[2:])
        return tensor

    @staticmethod
    def _init_audio2image(sample_rate=32000, n_fft=2048, win_length=2048,
                          hop_length=512, f_min=16., f_max=16386., pad=0,
                          n_mels=256, power=2., normalized=False, top_db=80.):
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


@add_to_catalog('attention', NN_CATALOG)
class AttentionNet(Net):
    def __init__(self, *args,  maxpool_loss=False, **kwargs):
        super(AttentionNet, self).__init__(*args, **kwargs)
        self.maxpool_loss = maxpool_loss
        self.maxpool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )

    def forward(self, wav_tensor, y=None):
        # wav_tensor: b, t
        if self.training:
            wav_tensor = self.batch_crop(wav_tensor)  # b, t

        spectrogram = self.audio2image(wav_tensor)  # b, m, t
        spectrogram = spectrogram.permute(0, 2, 1)  # b, t, m
        spectrogram = spectrogram[:, None, :, :]  # b, c, t, m

        if self.training:
            spectrogram = spectrogram.permute(0, 2, 1, 3)  # b, t, c, m
            spectrogram = self.batch_uncrop(spectrogram)

            spectrogram, y = self.mixup(spectrogram, y)

            spectrogram = self.batch_crop(spectrogram)
            spectrogram = spectrogram.permute(0, 2, 1, 3)  # b, c, t, m

        x = self.backbone(spectrogram)  # b, c, t, m
        if self.training:
            x = x.permute(0, 2, 1, 3)  # b, t, c, m
            x = self.batch_uncrop(x)
            x = x.permute(0, 2, 1, 3)  # b, c, t, m

        # average mel axis
        x = x.mean(axis=-1)

        attention_output = self.head(x)  # b, n_out
        logits = attention_output['logits']

        if y is not None:
            loss = self.loss(logits, y)
            if self.maxpool_loss:
                maxpool_logits = self.maxpool(attention_output['segmentwise_logits'])
                loss += 0.5 * self.loss(maxpool_logits, y)
        else:
            loss = None

        return {'loss': loss, 'logits': logits.sigmoid()}

    @staticmethod
    def _init_head(input_chs, output_len):
        head = Attention(input_chs, output_len, activation='linear')
        return head


@add_to_catalog('dropout_att', NN_CATALOG)
class DropOutAtt(AttentionNet):
    @staticmethod
    def _init_head(input_chs, output_len):
        head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv1d(input_chs, input_chs, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            Attention(input_chs, output_len, activation='linear')
        )
        return head


@add_to_catalog('att_focal', NN_CATALOG)
class FocalAttention(AttentionNet):
    def __init__(self, *args, **kwargs):
        super(FocalAttention, self).__init__(*args, **kwargs)
        self.loss = BCEFocalLoss()


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


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, activation='linear'):
        super().__init__()
        self.activation = activation
        self.attn = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.cla = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: b, c, t
        attn = torch.softmax(torch.tanh(self.attn(x)), dim=-1)  # b, c, t
        x = self.cla(x)  # b, c, t
        logits = torch.sum(x * attn, dim=-1)  # b, c
        return {'logits': logits, 'segmentwise_logits': x, 'attention': attn}


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss
