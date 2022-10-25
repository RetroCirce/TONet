"""
Ke Chen knutchen@ucsd.edu

Tone-Octave Network - data_generator file

This file contains the dataset and data generator classes

"""
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from util import index2centf
from feature_extraction import get_CenFreq

def reorganize(x, octave_res):
    n_order = []
    max_bin = x.shape[1]
    for i in range(octave_res):
        n_order += [j for j in range(i, max_bin, octave_res)]
    nx = [x[:,n_order[i],:] for i in range(x.shape[1])]
    nx = np.array(nx)
    nx = nx.transpose((1,0,2))
    return nx
     

class TONetTrainDataset(Dataset): 
    def __init__(self, data_list, config):
        self.config = config 
        self.cfp_dir = os.path.join(config.data_path,config.cfp_dir)
        self.f0_dir = os.path.join(config.data_path,"f0ref")
        self.data_list = data_list
        self.cent_f = np.array(get_CenFreq(config.startfreq, config.stopfreq, config.octave_res))
        # init data array
        self.data_cfp = []
        self.data_gd = []
        self.data_tcfp = []
        seg_frame = config.seg_frame
        shift_frame = config.shift_frame
        print("Data List:", data_list)
        with open(data_list, "r") as f:
            data_txt = f.readlines()
            data_txt = [d.split(".")[0] for d in data_txt]
        # data_txt = data_txt[:100]
        print("Song Size:", len(data_txt))
        # process cfp
        for i, filename in enumerate(tqdm(data_txt)):
            # file set
            cfp_file = os.path.join(self.cfp_dir, filename + ".npy")
            ref_file = os.path.join(self.f0_dir, filename + ".txt")
            # get raw cfp and freq
            temp_cfp = np.load(cfp_file, allow_pickle = True)
            # temp_cfp[0, :, :] = temp_cfp[1, :, :] * temp_cfp[2, :, :]
            temp_freq = np.loadtxt(ref_file)
            temp_freq = temp_freq[:,1]
            # check length
            if temp_freq.shape[0] > temp_cfp.shape[2]:
                temp_freq = temp_freq[:temp_cfp.shape[2]]
            else:
                temp_cfp = temp_cfp[:,:,:temp_freq.shape[0]]
            # build data
            for j in range(0, temp_cfp.shape[2], shift_frame): 
                bgnt = j
                endt = j + seg_frame
                temp_x = temp_cfp[:, :, bgnt:endt]
                temp_gd = index2centf(temp_freq[bgnt:endt], self.cent_f)
                
                if temp_x.shape[2] < seg_frame:
                    rl = temp_x.shape[2]
                    pad_x = np.zeros((temp_x.shape[0], temp_x.shape[1], seg_frame))
                    pad_gd = np.zeros((seg_frame))
                    pad_gd[:rl] = temp_gd
                    pad_x[:,:, :rl] = temp_x
                    temp_x = pad_x
                    temp_gd = pad_gd
                temp_tx = reorganize(temp_x[:], config.octave_res)
                self.data_tcfp.append(temp_tx)
                self.data_cfp.append(temp_x)
                self.data_gd.append(temp_gd)
        self.data_cfp = np.array(self.data_cfp)
        self.data_tcfp = np.array(self.data_tcfp)
        self.data_gd = np.array(self.data_gd)
        print("Total Datasize:", self.data_cfp.shape)
                       
    def __len__(self):
        return len(self.data_cfp)
    
    def __getitem__(self,index):
        temp_dict = {
            "cfp": self.data_cfp[index].astype(np.float32),
            "tcfp": np.zeros((1)), #self.data_tcfp[index].astype(np.float32),
            "gd": self.data_gd[index]
        }
        return temp_dict


class TONetTestDataset(Dataset): 
    def __init__(self, data_list, config):
        self.config = config 
        self.cfp_dir = os.path.join(config.data_path,config.cfp_dir)
        self.f0_dir = os.path.join(config.data_path,"f0ref")
        self.data_list = data_list
        self.cent_f = np.array(get_CenFreq(config.startfreq, config.stopfreq, config.octave_res))
        # init data array
        self.data_cfp = []
        self.data_gd = []
        self.data_len = []
        self.data_tcfp = []
        seg_frame = config.seg_frame
        shift_frame = config.shift_frame
        print("Data List:", data_list)
        with open(data_list, "r") as f:
            data_txt = f.readlines()
            data_txt = [d.split(".")[0] for d in data_txt]
        print("Song Size:", len(data_txt))
        # process cfp
        for i, filename in enumerate(tqdm(data_txt)):
            group_cfp = []
            group_gd = []
            group_tcfp = []
            # file set
            cfp_file = os.path.join(self.cfp_dir, filename + ".npy")
            ref_file = os.path.join(self.f0_dir, filename + ".txt")
            # get raw cfp and freq
            temp_cfp = np.load(cfp_file, allow_pickle = True)
            # temp_cfp[0, :, :] = temp_cfp[1, :, :] * temp_cfp[2, :, :]
            temp_freq = np.loadtxt(ref_file)
            temp_freq = temp_freq[:,1]
            self.data_len.append(len(temp_freq))
            # check length
            if temp_freq.shape[0] > temp_cfp.shape[2]:
                temp_freq = temp_freq[:temp_cfp.shape[2]]
            else:
                temp_cfp = temp_cfp[:,:,:temp_freq.shape[0]]
            # build data
            for j in range(0, temp_cfp.shape[2], shift_frame): 
                bgnt = j
                endt = j + seg_frame
                temp_x = temp_cfp[:, :, bgnt:endt]
                temp_gd = temp_freq[bgnt:endt]
                if temp_x.shape[2] < seg_frame:
                    rl = temp_x.shape[2]
                    pad_x = np.zeros((temp_x.shape[0], temp_x.shape[1], seg_frame))
                    pad_gd = np.zeros(seg_frame)
                    pad_gd[:rl] = temp_gd
                    pad_x[:,:, :rl] = temp_x
                    temp_x = pad_x
                    temp_gd = pad_gd
                temp_tx = reorganize(temp_x[:], config.octave_res)
                group_tcfp.append(temp_tx)
                group_cfp.append(temp_x)
                group_gd.append(temp_gd)
            group_tcfp = np.array(group_tcfp)
            group_cfp = np.array(group_cfp)
            group_gd = np.array(group_gd)
            self.data_tcfp.append(group_tcfp)
            self.data_cfp.append(group_cfp)
            self.data_gd.append(group_gd)
                       
    def __len__(self):
        return len(self.data_cfp)
    
    def __getitem__(self,index):
        temp_dict = {
            "cfp": self.data_cfp[index].astype(np.float32),
            "tcfp": self.data_tcfp[index].astype(np.float32),
            "gd": self.data_gd[index],
            "length": self.data_len[index]
        }
        return temp_dict
