"""
Ke Chen knutchen@ucsd.edu

Tone-Octave Network - model

This file contains the TONet core code

"""
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from util import melody_eval, freq2octave, freq2tone, tofreq
from .attention_layer import CombineLayer, PositionalEncoding
from feature_extraction import get_CenFreq


class TONet(pl.LightningModule):
    """
    Args:
        mode: ["disable", "enable"]
    """
    def __init__(self, l_model, r_model, config, loss_func, mode = "single"):
        super().__init__()
        self.config = config
        # l_model for original-CFP
        self.l_model = l_model
        # r_model for Tone-CFP
        self.r_model = r_model
        self.mode = mode
        self.centf = np.array(get_CenFreq(config.startfreq, config.stopfreq, config.octave_res))
        self.centf[0] = 0
        self.loss_func = loss_func
        self.max_metric = np.zeros((3, 6))
        if self.mode == "all" or self.mode == "tcfp":
            assert r_model is not None, "Enabling TONet needs two-branch models!"


        self.gru_dim = 512
        self.attn_dim = 2048
        # define hyperparameter
        if self.mode == "tcfp":
            self.sp_dim = self.config.freq_bin * 2
            self.linear_dim = self.config.freq_bin * 2
        elif self.mode == "spl":
            self.sp_dim = self.config.freq_bin
            self.linear_dim = self.gru_dim * 2
        elif self.mode == "spat":
            self.sp_dim = self.config.freq_bin
            self.linear_dim = self.attn_dim
        elif self.mode == "all":
            self.sp_dim = self.config.freq_bin * 2
            self.linear_dim = self.attn_dim

        # Network Architecture 
        if self.mode == "spl":
            self.tone_gru = nn.Linear(self.sp_dim, self.linear_dim)
            # nn.GRU(
                # self.sp_dim, self.gru_dim, 1,
                # batch_first=True, bidirectional=True
            # )
            self.octave_gru = nn.Linear(self.sp_dim, self.linear_dim)
            # nn.GRU(
            #     self.sp_dim, self.gru_dim, 1,
            #     batch_first=True, bidirectional=True
            # )
        elif self.mode == "spat" or self.mode == "all":
            self.tone_in = nn.Linear(self.sp_dim, self.attn_dim)
            self.tone_posenc = PositionalEncoding(self.attn_dim, n_position = self.config.seg_frame)
            self.tone_dropout = nn.Dropout(p = 0.2)
            self.tone_norm = nn.LayerNorm(self.attn_dim, eps = 1e-6)
            self.tone_attn = nn.ModuleList([
                CombineLayer(self.attn_dim, self.attn_dim * 2, 8, 
                self.attn_dim // 8, self.attn_dim // 8, dropout = 0.2)
                for _ in range(2)]
            )
            self.octave_in = nn.Linear(self.sp_dim, self.attn_dim)
            self.octave_posenc = PositionalEncoding(self.attn_dim, n_position = self.config.seg_frame)
            self.octave_dropout = nn.Dropout(p = 0.2)
            self.octave_norm = nn.LayerNorm(self.attn_dim, eps = 1e-6)
            self.octave_attn = nn.ModuleList([
                CombineLayer(self.attn_dim, self.attn_dim * 2, 8, 
                self.attn_dim // 8, self.attn_dim // 8, dropout = 0.2)
                for _ in range(2)]
            )
        if self.mode != "single" and self.mode != "tcfp":
            self.tone_linear = nn.Sequential(
                nn.Linear(self.linear_dim, 512),
                nn.Dropout(p = 0.2),
                nn.SELU(),
                nn.Linear(512, 128),
                nn.Dropout(p = 0.2),
                nn.SELU(),
                nn.Linear(128, self.config.tone_class),
                nn.Dropout(p = 0.2),
                nn.SELU()
            )
            self.octave_linear = nn.Sequential(
                nn.Linear(self.linear_dim, 256),
                nn.Dropout(p = 0.2),
                nn.SELU(),
                nn.Linear(256, 64),
                nn.Dropout(p = 0.2),
                nn.SELU(),
                nn.Linear(64, self.config.octave_class),
                nn.Dropout(p = 0.2),
                nn.SELU()
            )
            self.tone_bm = nn.Sequential(
                nn.Linear(2, 1),
                nn.SELU()
            )
            self.octave_bm = nn.Sequential(
                nn.Linear(2, 1),
                nn.SELU()
            )
            # [bs, 361 + 13 + 9, 128]
            self.tcfp_linear = nn.Sequential(
                nn.Conv1d(self.config.freq_bin * 2, self.config.freq_bin,
                5, padding=2),
                nn.SELU()
            )
            self.tcfp_bm = nn.Sequential(
                nn.Conv1d(2,1,5,padding=2),
                nn.SELU()
            )
            self.final_linear = nn.Sequential(
                nn.Conv1d(
                    self.config.tone_class + self.config.octave_class + self.config.freq_bin + 3,
                    self.config.freq_bin, 5, padding=2),
                nn.SELU()
            )
        elif self.mode == "tcfp":
            self.final_linear = nn.Sequential(
                nn.Linear(self.linear_dim, self.config.freq_bin),
                nn.SELU()
            )
            self.final_bm = nn.Sequential(
                nn.Linear(2, 1),
                nn.SELU()
            )
    """
    Args:
        x: [bs, 3, freuqncy_bin, time_frame]
    """
    def tone_decoder(self, tone_feature):
        if self.mode == "all" or self.mode == "spat":
            tone_h = self.tone_dropout(self.tone_posenc(self.tone_in(tone_feature)))
            tone_h = self.tone_norm(tone_h)
            for tone_layer in self.tone_attn:
                tone_h, tone_weight = tone_layer(tone_h, slf_attn_mask = None)
            tone_prob = self.tone_linear(tone_h)
            tone_prob = tone_prob.permute(0, 2, 1).contiguous()
        elif self.mode == "spl":
            tone_h = self.tone_gru(tone_feature)
            tone_prob = self.tone_linear(tone_h)
            tone_prob = tone_prob.permute(0, 2, 1).contiguous()
        return tone_prob

    def octave_decoder(self, octave_feature):
        if self.mode == "all" or self.mode == "spat":
            octave_h = self.octave_dropout(self.octave_posenc(self.octave_in(octave_feature)))
            octave_h = self.octave_norm(octave_h)
            for octave_layer in self.octave_attn:
                octave_h, octave_weight = octave_layer(octave_h, slf_attn_mask = None)
            octave_prob = self.octave_linear(octave_h)
            octave_prob = octave_prob.permute(0, 2, 1).contiguous()
        elif self.mode == "spl":
            octave_h = self.octave_gru(octave_feature)
            octave_prob = self.octave_linear(octave_h)
            octave_prob = octave_prob.permute(0, 2, 1).contiguous()
        return octave_prob


    def forward(self, x, tx = None):
        if self.mode == "single":
            output, _ = self.l_model(x)
            return output
        elif self.mode == "all":
            _, output_l = self.l_model(x)
            _, output_r = self.r_model(tx)
            bm_l = output_l[:, :, 0, :].unsqueeze(dim = 2)
            output_l = output_l[:,:, 1:,:]
            bm_r = output_r[:, :, 0, :].unsqueeze(dim = 2)
            output_r = output_r[:,:, 1:,:]
            feature_agg = torch.cat((output_l, output_r), dim = 2)
            feature_agg = feature_agg.squeeze(dim = 1)
            feature_agg_mi = self.tcfp_linear(feature_agg) # [bs, 360, 128]
            bm_agg = torch.cat((bm_l, bm_r), dim = 2)
            bm_agg = bm_agg.squeeze(dim = 1)
            bm_agg_mi = self.tcfp_bm(bm_agg)
            bm_agg = bm_agg.permute(0,2,1)
            tone_feature = feature_agg.permute(0,2,1).contiguous()
            octave_feature = feature_agg.permute(0,2,1).contiguous()
            tone_prob = self.tone_decoder(tone_feature)
            octave_prob = self.octave_decoder(octave_feature)

            tone_bm = self.tone_bm(bm_agg)
            octave_bm = self.octave_bm(bm_agg)
            tone_bm = tone_bm.permute(0,2,1)
            octave_bm = octave_bm.permute(0,2,1)

            tone_prob = torch.cat((tone_prob, tone_bm), dim = 1)
            octave_prob = torch.cat((octave_prob, octave_bm), dim = 1)
            
            final_feature = torch.cat((tone_prob, octave_prob, feature_agg_mi, bm_agg_mi), dim = 1)
            final_feature = self.final_linear(final_feature)
            final_feature = torch.cat((bm_agg_mi, final_feature), dim=1)
            final_feature = nn.Softmax(dim = 1)(final_feature)
            tone_prob = nn.Softmax(dim = 1)(tone_prob)
            octave_prob = nn.Softmax(dim = 1)(octave_prob)
            return tone_prob, octave_prob, final_feature
        elif self.mode == "tcfp":
            _, output_l = self.l_model(x)
            _, output_r = self.r_model(tx)
            bm_l = output_l[:, :, 0, :].unsqueeze(dim = 2)
            output_l = output_l[:,:, 1:,:]
            bm_r = output_r[:, :, 0, :].unsqueeze(dim = 2)
            output_r = output_r[:,:, 1:,:]
            feature_agg = torch.cat((output_l, output_r), dim = 2)
            feature_agg = feature_agg.permute(0, 1, 3, 2)
            bm_agg = torch.cat((bm_l, bm_r), dim = 2)
            bm_agg = bm_agg.permute(0, 1, 3, 2)
            final_x = self.final_linear(feature_agg)
            final_bm = self.final_bm(bm_agg)
            final_x = final_x.permute(0,1,3,2)
            final_bm = final_bm.permute(0,1,3,2)
            final_output = nn.Softmax(dim = 2)(torch.cat((final_bm, final_x), dim = 2))
            return final_output
        elif self.mode == "spl" or self.mode == "spat":
            _, output_l = self.l_model(x)
            bm_l = output_l[:, :, 0, :].unsqueeze(dim = 2)
            output_l = output_l[:,:, 1:,:]
            feature_agg = output_l
            feature_agg = feature_agg.squeeze(dim = 1)
            bm_agg = bm_l
            bm_agg = bm_agg.squeeze(dim = 1)
            tone_feature = feature_agg.permute(0,2,1).contiguous()
            octave_feature = feature_agg.permute(0,2,1).contiguous()
            tone_prob = self.tone_decoder(tone_feature)
            octave_prob = self.octave_decoder(octave_feature)
            tone_bm = bm_agg
            octave_bm = bm_agg

            tone_prob = torch.cat((tone_prob, tone_bm), dim = 1)
            octave_prob = torch.cat((octave_prob, octave_bm), dim = 1)
            
            final_feature = torch.cat((tone_prob, octave_prob, feature_agg, bm_agg), dim = 1)
            final_feature = self.final_linear(final_feature)
            final_feature = torch.cat((bm_agg, final_feature), dim=1)
            final_feature = nn.Softmax(dim = 1)(final_feature)
            tone_prob = nn.Softmax(dim = 1)(tone_prob)
            octave_prob = nn.Softmax(dim = 1)(octave_prob)
            return tone_prob, octave_prob, final_feature
    """
    Args:
        batch: {
            "cfp": [bs, 3, frequency_bin, time_frame],
            "gd": [bs, time_frame]
        }
    """
    def training_step(self, batch, batch_idx):
        device_type = next(self.parameters()).device
        cfps = batch["cfp"]
        tcfps = batch["tcfp"]
        gds = batch["gd"]
        if self.mode == "single":
            gd_maps = torch.zeros((cfps.shape[0], cfps.shape[-2] + 1, cfps.shape[-1])).to( device_type)
            for i in range(len(gds)):
                gd_maps[i, gds[i].long(), torch.arange(gds.shape[-1])] = 1.0
            output = self(cfps)
            output = torch.squeeze(output, dim = 1)
            loss = self.loss_func(output, gd_maps)
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        elif self.mode == "all":
            # from pure pitch estimation
            gd_maps = torch.zeros((cfps.shape[0], cfps.shape[-2] + 1, cfps.shape[-1])).to( device_type)
            tone_maps = torch.zeros((cfps.shape[0], self.config.tone_class + 1, cfps.shape[-1])).to(device_type)
            octave_maps = torch.zeros((cfps.shape[0], self.config.octave_class + 1, cfps.shape[-1])).to(device_type)
            tone_index = ((gds % 60) * self.config.tone_class / 60).long()
            octave_index = (gds // 60 + 2).long()
            tone_index[gds < 1.0] = self.config.tone_class
            octave_index[gds < 1.0] = self.config.octave_class
            for i in range(len(tone_maps)):
                tone_maps[i, tone_index[i], torch.arange(gds.shape[-1])] = 1.0
                octave_maps[i, octave_index[i], torch.arange(gds.shape[-1])] = 1.0
                gd_maps[i, gds[i].long(), torch.arange(gds.shape[-1])] = 1.0
            tone_prob, octave_prob, final_prob = self(cfps, tcfps)
            pred_map = torch.cat((tone_prob, octave_prob , final_prob), dim = 1)
            gd_map = torch.cat([tone_maps, octave_maps, gd_maps], dim = 1)
            loss = self.loss_func(pred_map, gd_map)
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        elif self.mode == "tcfp":
            gd_maps = torch.zeros((cfps.shape[0], cfps.shape[-2] + 1, cfps.shape[-1])).to( device_type)
            for i in range(len(gds)):
                gd_maps[i, gds[i].long(), torch.arange(gds.shape[-1])] = 1.0
            output = self(cfps, tcfps)
            output = torch.squeeze(output, dim = 1)
            loss = self.loss_func(output, gd_maps)
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        elif self.mode == "spl" or self.mode == "spat":
            # from pure pitch estimation
            gd_maps = torch.zeros((cfps.shape[0], cfps.shape[-2] + 1, cfps.shape[-1])).to( device_type)
            tone_maps = torch.zeros((cfps.shape[0], self.config.tone_class + 1, cfps.shape[-1])).to(device_type)
            octave_maps = torch.zeros((cfps.shape[0], self.config.octave_class + 1, cfps.shape[-1])).to(device_type)
            tone_index = ((gds % 60) * self.config.tone_class / 60).long()
            octave_index = (gds // 60 + 2).long()
            tone_index[gds < 1.0] = self.config.tone_class
            octave_index[gds < 1.0] = self.config.octave_class
            for i in range(len(tone_maps)):
                tone_maps[i, tone_index[i], torch.arange(gds.shape[-1])] = 1.0
                octave_maps[i, octave_index[i], torch.arange(gds.shape[-1])] = 1.0
                gd_maps[i, gds[i].long(), torch.arange(gds.shape[-1])] = 1.0
            tone_prob, octave_prob, final_prob = self(cfps)
            pred_map = torch.cat((tone_prob, octave_prob , final_prob), dim = 1)
            gd_map = torch.cat([tone_maps, octave_maps, gd_maps], dim = 1)
            loss = self.loss_func(pred_map, gd_map)
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    

    def write_prediction(self, pred, filename):
        time_frame = np.arange(len(pred)) * 0.01
        with open(filename, "w") as f:
            for i in range(len(time_frame)):
                f.write(str(np.round(time_frame[i], 4)) + "\t" + str(pred[i]) + "\n")


    def validation_step(self, batch, batch_idx, dataset_idx):
        device_type = next(self.parameters()).device
        mini_batch = self.config.batch_size
        cfps = batch["cfp"][0]
        tcfps = batch["tcfp"][0]
        gds = batch["gd"][0]
        lens = batch["length"][0]
        
        if self.mode == "single":
            output = []
            for i in range(0, len(cfps), mini_batch):
                temp_cfp = torch.from_numpy(cfps[i:i + mini_batch]).to(device_type)
                temp_output = self(temp_cfp)
                temp_output = torch.squeeze(temp_output, dim = 1)
                temp_output = temp_output.detach().cpu().numpy()
                output.append(temp_output)
            output = np.concatenate(np.array(output),axis = 0)
            return [
                output, 
                gds, 
                lens
            ]
        elif self.mode == "all":
            # output_tone = []
            # output_octave = []
            output = []
            for i in range(0, len(cfps), mini_batch):
                temp_cfp = torch.from_numpy(cfps[i:i + mini_batch]).to(device_type)
                temp_tcfp = torch.from_numpy(tcfps[i:i + mini_batch]).to(device_type)
                _, _, temp_output = self(temp_cfp, temp_tcfp)
                temp_output = temp_output.detach().cpu().numpy()
                output.append(temp_output)
            output = np.concatenate(output,axis = 0)
            return [

                output,
                gds, 
                lens
            ]
        elif self.mode == "tcfp":
            output = []
            for i in range(0, len(cfps), mini_batch):
                temp_cfp = torch.from_numpy(cfps[i:i + mini_batch]).to(device_type)
                temp_tcfp = torch.from_numpy(tcfps[i:i + mini_batch]).to(device_type)
                temp_output = self(temp_cfp, temp_tcfp)
                temp_output = torch.squeeze(temp_output, dim = 1)
                temp_output = temp_output.detach().cpu().numpy()
                output.append(temp_output)
            output = np.concatenate(np.array(output),axis = 0)
            return [
                output, 
                gds, 
                lens
            ]
        elif self.mode == "spl" or self.mode == "spat":
            # output_tone = []
            # output_octave = []
            output = []
            for i in range(0, len(cfps), mini_batch):
                temp_cfp = torch.from_numpy(cfps[i:i + mini_batch]).to(device_type)
                _,_ , temp_output = self(temp_cfp)
                temp_output = temp_output.detach().cpu().numpy()
                output.append(temp_output)
            output = np.concatenate(output,axis = 0)
            return [
                output,
                gds, 
                lens
            ]

    def validation_epoch_end(self, validation_step_outputs):
        if self.mode == "single" or self.mode == "tcfp":
            for i, dataset_d in enumerate(validation_step_outputs):  
                metric = np.array([0.,0.,0.,0.,0.,0.])  
                preds = []
                gds = []
                for d in dataset_d:
                    pred, gd, rl = d
                    pred = np.argmax(pred, axis = 1)
                    pred = np.concatenate(pred, axis = 0)
                    pred = self.centf[pred]
                    gd = np.concatenate(gd, axis = 0)
                    preds.append(pred)
                    gds.append(gd)
                preds = np.concatenate(preds, axis = 0)
                gds = np.concatenate(gds, axis = 0)
                metric = melody_eval(preds, gds)
                self.print("\n")
                self.print("Dataset ", i, " OA:", metric[-1])
                if metric[-1] > self.max_metric[i, -1]:
                    for j in range(len(self.max_metric[i])):
                        self.max_metric[i,j] = metric[j]
                        self.max_metric[i,j] = metric[j]
                    torch.save(self.state_dict(), "model_backup/bestk_" + str(i) + ".ckpt")
                self.print("Best ",i,":", self.max_metric[i])
        elif self.mode == "all" or self.mode == "spl" or self.mode == "spat":
            for i, dataset_d in enumerate(validation_step_outputs):    
                metric = np.array([0.,0.,0.,0.,0.,0.])  
                preds = []
                gds = []
                for d in dataset_d:
                    pred, gd, rl = d
                    pred = np.argmax(pred, axis = 1)
                    pred = np.concatenate(pred, axis = 0)
                    pred = self.centf[pred]
                    gd = np.concatenate(gd, axis = 0)
                    preds.append(pred)
                    gds.append(gd)
                preds = np.concatenate(preds, axis = 0)
                gds = np.concatenate(gds, axis = 0)
                metric = melody_eval(preds, gds)
                self.print("\n")
                self.print("Dataset ", i, " OA:", metric[-1])
                if metric[-1] > self.max_metric[i, -1]:
                    for j in range(len(self.max_metric[i])):
                        self.max_metric[i,j] = metric[j]
                        self.max_metric[i,j] = metric[j]
                    torch.save(self.state_dict(), "model_backup/bestk_" + str(i) + ".ckpt")
                self.print("Best ",i,":", self.max_metric[i])
    def test_step(self, batch, batch_idx, dataset_idx):
        return self.validation_step(batch, batch_idx, dataset_idx)

    def test_epoch_end(self, test_step_outputs):
        self.validation_epoch_end(test_step_outputs)
        # for i, dataset_d in enumerate(test_step_outputs):  
        #     for j, d in enumerate(dataset_d):
        #         pred, _, rl = d
        #         pred = np.argmax(pred, axis = 1)
        #         pred = np.concatenate(pred, axis = 0)[:rl]
        #         pred = self.centf[pred]
                # self.write_prediction(pred, "prediction/" + str(i) + "_" + str(j) + ".txt")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        def lr_foo(epoch):
            if epoch < 5:
                # warm up lr
                lr_scale = 0.5
            else:
                lr_scale = 0.5 * 0.98 ** (epoch - 5)

            return lr_scale

        if self.mode == "single" or self.mode == "tcfp":
            return optimizer
        elif self.mode == "all" or self.mode == "spl" or self.mode == "spat":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_foo
            )
            return [optimizer], [scheduler]
