import os
import random
import numpy as np
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import config
from data_generator import TONetTrainDataset, TONetTestDataset
from model.msnet import MSnet
from model.tonet import TONet
from model.multi_dr import MLDRnet
from model.ftanet import FTAnet

from util import pos_weight, tonpy_fn

def train():
    train_dataset = TONetTrainDataset(
        data_list = config.train_file,
        config = config
    )
    train_dataloader = DataLoader(
        dataset = train_dataset,
        shuffle = True,
        num_workers = config.n_workers,
        batch_size = config.batch_size
    )
    test_datasets = [
        TONetTestDataset(
            data_list = d,
            config = config
        ) for d in config.test_file
    ]
    test_dataloaders = [
        DataLoader(
            dataset = d,
            shuffle = False,
            batch_size = 1,
            collate_fn=tonpy_fn
        ) for d in test_datasets
    ]
    # pw = pos_weight(train_dataset.data_gd, 321)
    loss_func = nn.BCELoss()
    # loss_func = nn.BCEWithLogitsLoss(
        # pos_weight = pw
    # )
    # me_model = FTAnet(freq_bin = config.freq_bin, time_segment=config.seg_frame)
    # me_model_r = FTAnet(freq_bin = config.freq_bin, time_segment=config.seg_frame)
    me_model = MSnet()
    me_model_r = MSnet()
    if config.ablation_mode == "single" or config.ablation_mode == "spl" or config.ablation_mode == "spat":
        me_model_r = None #MSnet()
    # me_model = MLDRnet()
    # me_model.load_state_dict(torch.load("data/MSnet_vocal.ckpt", map_location={'cuda:2':'cuda:{}'.format(0)}))
    model = TONet(
        l_model = me_model,
        r_model = me_model_r,
        config = config,
        loss_func = loss_func,
        mode = config.ablation_mode
    )
    trainer = pl.Trainer(
        deterministic = True,
        gpus = 1,
        checkpoint_callback = False,
        max_epochs = config.max_epoch,
        auto_lr_find = True,
        sync_batchnorm=True,
        # check_val_every_n_epoch = 1,
        val_check_interval = 0.25,
    )
    # trainer.test(model, test_dataloaders)
    trainer.fit(model, train_dataloader, test_dataloaders)
    

def test():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "TONET for Singing Melody Extraction")
    subparsers = parser.add_subparsers(dest = "mode")
    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")
    args = parser.parse_args()
    pl.seed_everything(config.random_seed)
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()



"""
ADC

MIREX
83.44 8.82 76.79 76.97 81.97

MELODY
60.07 11.85 51.66 52.79 69.01
"""