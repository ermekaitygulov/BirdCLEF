from typing import Dict, Type

import torch
from torch.nn import Module
import torch.nn.functional as F


class BaseModel(Module):
    def __init__(self, input_dim, output_dim, device, trg_vocab, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kwargs = kwargs
        self.device = device
        self.trg_vocab = trg_vocab
        self.pad_idx = trg_vocab.stoi['<pad>']
        self.sos_idx = trg_vocab.stoi['<sos>']
        self.eos_idx = trg_vocab.stoi['<eos>']

    def gen_translate(self, src, trg, greedy=True):
        output_logits, output_seq = self(src, trg, 1., greedy=greedy)
        output_prob = F.log_softmax(output_logits, dim=-1)
        return output_prob, output_seq

    def load(self, path):
        with open(path, "rb") as fp:
            state_dict = torch.load(fp, map_location='cpu')
            self.load_state_dict(state_dict)


NN_CATALOG: Dict[str, Type[BaseModel]] = {}
