"""
Ke Chen knutchen@ucsd.edu

Tone-Octave Network - main file

This file contains the main script

"""
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
from model.mcdnn import MCDNN
from model.spectnt import SpecTNT

from util import tonpy_fn


def train():
    train_dataset = TONetTrainDataset(
        data_list = config.train_file,
        config = config
    )
    train_dataloader = DataLoader(
        dataset = train_dataset,
        shuffle = True,
        num_workers = config.n_workers,
        batch_size = config.batch_size,
        drop_last=True
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
    loss_func = nn.BCELoss()

    if config.model_type == "MCDNN":
        me_model = MCDNN()
        me_model_r = MCDNN()
    elif config.model_type == "MLDRNet":
        me_model = MLDRnet()
        me_model_r = MLDRnet()
    elif config.model_type == "FTANet":
        me_model = FTAnet(freq_bin = config.freq_bin, time_segment=config.seg_frame)
        me_model_r = FTAnet(freq_bin = config.freq_bin, time_segment=config.seg_frame)
    elif config.model_type == "MSNet": # MSNet
        me_model = MSnet() 
        me_model_r = MSnet()
    else: # SpecTNT
        me_model = SpecTNT(
            spec_size=384,
            patch_size=4,
            in_chans=3,
            num_classes=360,
            window_size=12,
            config=config,
            depths=[2,2,18,2],
            embed_dim=128,
            patch_stride=(4,4),
            num_heads=[4,8,16,32]
        )
        me_model_r = SpecTNT(
            spec_size=384,
            patch_size=4,
            in_chans=3,
            num_classes=360,
            window_size=12,
            config=config,
            depths=[2,2,18,2],
            embed_dim=128,
            patch_stride=(4,4),
            num_heads=[4,8,16,32]
        ) 
    if config.spectnt_ckpt is not None:
        ckpt = torch.load(config.spectnt_ckpt, map_location="cpu")
        # load pretrain model
        ckpt = ckpt["model"]
        found_parameters = []
        unfound_parameters = []
        model_params = dict(me_model.state_dict())

        for key in model_params:
            m_key = key # .replace("sed_model.", "")
            if m_key in ckpt:
                # if m_key == "patch_embed.proj.weight":
                    # ckpt[m_key] = torch.mean(ckpt[m_key], dim = 1, keepdim = True)
                    # print(ckpt[m])
                if m_key == "head.weight" or m_key == "head.bias":
                    ckpt.pop(m_key)
                    unfound_parameters.append(key)
                    continue
                assert model_params[key].shape==ckpt[m_key].shape, "%s is not match, %s vs. %s" %(key, str(model_params[key].shape), str(ckpt[m_key].shape))
                found_parameters.append(key)
                ckpt[key] = ckpt.pop(m_key)
            else:
                unfound_parameters.append(key)
        print("pretrain param num: %d \t wrapper param num: %d"%(len(found_parameters), len(ckpt.keys())))
        print("unfound parameters: ", unfound_parameters)
        me_model.load_state_dict(ckpt, strict = False)
        me_model_r.load_state_dict(ckpt, strict = False)

    if config.ablation_mode == "single" or config.ablation_mode == "spl" or config.ablation_mode == "spat":
        me_model_r = None 

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
        val_check_interval = 0.1,
        # num_sanity_val_steps=0
    )
    trainer.fit(model, train_dataloader, test_dataloaders)
    

def test():
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
    loss_func = nn.BCELoss()

    if config.model_type == "MCDNN":
        me_model = MCDNN()
        me_model_r = MCDNN()
    elif config.model_type == "MLDRNet":
        me_model = MLDRnet()
        me_model_r = MLDRnet()
    elif config.model_type == "FTANet":
        me_model = FTAnet(freq_bin = config.freq_bin, time_segment=config.seg_frame)
        me_model_r = FTAnet(freq_bin = config.freq_bin, time_segment=config.seg_frame)
    elif config.model_type == "MSNet": # MSNet
        me_model = MSnet() 
        me_model_r = MSnet()
    else: # SpecTNT
        me_model = SpecTNT(
            spec_size=384,
            patch_size=4,
            in_chans=3,
            num_classes=360,
            window_size=12,
            config=config,
            depths=[2,2,18,2],
            embed_dim=128,
            patch_stride=(4,4),
            num_heads=[4,8,16,32]
        )
        me_model_r = SpecTNT(
            spec_size=384,
            patch_size=4,
            in_chans=3,
            num_classes=360,
            window_size=12,
            config=config,
            depths=[2,2,18,2],
            embed_dim=128,
            patch_stride=(4,4),
            num_heads=[4,8,16,32]
        ) 
    if config.ablation_mode == "single" or config.ablation_mode == "spl" or config.ablation_mode == "spat":
        me_model_r = None 
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
    # load the checkpoint
    ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt)
    trainer.test(model, test_dataloaders)


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

