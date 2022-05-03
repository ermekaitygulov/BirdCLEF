"""script for train"""
from argparse import ArgumentParser

import torch
import wandb
import yaml

from experiment import EXPERIMENT_CATALOG

WANDB_GLOBAL = dict(
    entity='ermekaitygulov',
    anonymous='allow',
    project='Bird-CLEF'
)


def parser_conf(parser):
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and logging).",
                        default="baseline")
    parser.add_argument("--config", "-c")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.set_defaults(callback=train_callback)
    return parser


def train_callback(args):
    experiment_name = args.name
    experiment_config = read_config(args.config)
    if args.wandb:
        init_wandb(args.name, experiment_config)

    device = configure_device(args.gpu)
    experiment = EXPERIMENT_CATALOG[experiment_name](experiment_config, device)
    experiment.train()
    experiment.test()


def read_config(config_path):
    with open(config_path) as fin:
        config = yaml.safe_load(fin)
    return config


def configure_device(gpu_flag):
    device = torch.device('cuda' if torch.cuda.is_available() and gpu_flag else 'cpu')
    return device


def init_wandb(experiment_name, config):
    name = f'{experiment_name}_{wandb.util.generate_id()}'
    wandb.init(name=name, config=config, group=experiment_name, **WANDB_GLOBAL)


if __name__ == '__main__':
    train_parser = ArgumentParser(__doc__)
    train_parser = parser_conf(train_parser)
    train_args = train_parser.parse_args()
    train_args.callback(train_args)