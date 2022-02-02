"""
Ke Chen knutchen@ucsd.edu

Tone-Octave Network - utils file

This file contains useful common methods

"""
import os
import numpy as np
import torch
import mir_eval
import config

def index2centf(seq, centfreq):
    centfreq[0] = 0
    re = np.zeros(len(seq))
    for i in range(len(seq)):
        for j in range(len(centfreq)):
            if seq[i] < 0.1:
                re[i] = 0
                break
            elif centfreq[j] > seq[i]:
                re[i] = j
                break
    return re  


def freq2octave(freq):
    if freq < 1.0 or freq > 2050:
        return config.octave_class
    else:
        return int(np.round(69 + 12 * np.log2(freq/440)) // 12) 

def freq2tone(freq):
    if freq < 1.0 or freq > 2050:
        return config.tone_class
    else:
        return int(np.round(69 + 12 * np.log2(freq/440)) % 12) 

def tofreq(tone, octave):
    if tone >= config.tone_class or octave >= config.octave_class or octave < 2:
        return 0.0
    else:
        return 440 * 2 ** ((12 * octave + tone * 12 / config.tone_class - 69) / 12)


def pos_weight(data, freq_bins):
    frames = data.shape[-1]
    non_vocal = float(len(data[data == 0]))
    vocal = float(data.size - non_vocal)
    z = np.zeros((freq_bins, frames))
    z[1:,:] += (non_vocal / vocal)
    z[0,:] += vocal / non_vocal
    print(non_vocal, vocal)
    return torch.from_numpy(z).float()

def freq2octave(freq):
    if freq < 1.0 or freq > 1990: 
        return 0
    pitch = round(69 + 12 * np.log2(freq / 440))
    return int(pitch // 12)

def compute_roa(pred, gd):
    pred = pred[gd > 0.1]
    gd = gd[gd > 0.1]
    pred = np.array([freq2octave(d) for d in pred])
    gd = np.array([freq2octave(d) for d in gd])
    return np.sum(pred == gd) / len(pred)


def melody_eval(pred, gd):
    ref_time = np.arange(len(gd)) * 0.01
    ref_freq = gd

    est_time = np.arange(len(pred)) * 0.01
    est_freq = pred

    output_eval = mir_eval.melody.evaluate(ref_time,ref_freq,est_time,est_freq)
    VR = output_eval['Voicing Recall']*100.0 
    VFA = output_eval['Voicing False Alarm']*100.0
    RPA = output_eval['Raw Pitch Accuracy']*100.0
    RCA = output_eval['Raw Chroma Accuracy']*100.0
    ROA = compute_roa(est_freq, ref_freq) * 100.0
    OA = output_eval['Overall Accuracy']*100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, ROA, OA])
    return eval_arr

def tonpy_fn(batch):
    dict_key = batch[0].keys()
    output_batch = {}
    for dk in dict_key:
        output_batch[dk] = np.array([d[dk] for d in batch])
    return output_batch