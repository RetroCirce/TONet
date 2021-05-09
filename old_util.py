import numpy as np
import torch
import time

import mir_eval

LEN_SEG = 64

def melody_eval(ref_time, ref_freq, est_time, est_freq):

    output_eval = mir_eval.melody.evaluate(ref_time,ref_freq,est_time,est_freq)
    VR = output_eval['Voicing Recall']*100.0 
    VFA = output_eval['Voicing False Alarm']*100.0
    RPA = output_eval['Raw Pitch Accuracy']*100.0
    RCA = output_eval['Raw Chroma Accuracy']*100.0
    OA = output_eval['Overall Accuracy']*100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    return eval_arr

def pred2res(pred):
    '''
    Convert the output of model to the result
    '''
    pred = np.array(pred)
    pred_freq = pred.argmax(axis=1)
    pred_freq[pred_freq > 0] = 31 * 2 ** (pred_freq[pred_freq > 0] / 60)
    return pred_freq

def pred2resT(pred, threshold=0.1):
    '''
    Convert the output of model to the result
    '''
    pred = np.array(torch.softmax(pred, dim=1))
    pred_freq = pred.argmax(axis=1)
    pred_freq[pred.max(axis=1) <= 0.1] = 0
    pred_freq[pred_freq > 0] = 31 * 2 ** (pred_freq[pred_freq > 0] / 60)
    return pred_freq

def y2res(y):
    '''
    Convert the label to the result
    '''
    y = np.array(y)
    y[y > 0] = 31 * 2 ** (y[y > 0] / 60)
    return y
    
def convert_y(y):
    y[y > 0] = torch.round(torch.log2(y[y > 0] / 31) * 60)
    return y.long()

def load_list(path, mode):
    if mode == 'test':
        f = open(path, 'r')
    elif mode == 'train':
        f = open(path, 'r')
    else:
        raise Exception("mdoe must be 'test' or 'train'")
    data_list = []
    for line in f.readlines():
        data_list.append(line.strip())
    if mode == 'test':
        print("{:d} test files: ".format(len(data_list)), data_list)
    else:
        print("{:d} train files: ".format(len(data_list)), data_list)
    return data_list

def load_train_data(path):
    tick = time.time()
    train_list = load_list(path, mode='train')
    X, y = [], []
    num_seg = 0
    for i in range(len(train_list)):
        print('({:d}/{:d}) Loading data: '.format(i+1, len(train_list)), train_list[i])
        X_data, y_data = load_data(train_list[i])
        y_data[y_data>320] = 320
        seg = X_data.size(0)
        num_seg += seg
        X.append(X_data)
        y.append(y_data)
        print('({:d}/{:d})'.format(i+1, len(train_list)), train_list[i], 'loaded: ', '{:2d} segments'.format(seg))
    print("Training data loaded in {:.2f}(s): {:d} segments".format(time.time() - tick, num_seg))
    return torch.cat(X, dim=0), torch.cat(y, dim=0)

def load_data(fp):
    '''
    X: (N, C, F, T)
    y: (N, T)
    '''
    X = np.load('/home/kechen/Research/ISMIR-2021-TONET/data/cfp/' + fp)
    L = X.shape[2]
    num_seg = L // LEN_SEG
    X = torch.tensor([X[:, :, LEN_SEG*i:LEN_SEG*i+LEN_SEG] for i in range(num_seg)], dtype=torch.float32)
    #X = (X[:, 1] * X[:, 2]).unsqueeze(dim=1)

    f = open('/home/kechen/Research/ISMIR-2021-TONET/data/f0ref/' + fp.replace('.npy', '') + '.txt')
    y = []
    for line in f.readlines():
        y.append(float(line.strip().split()[1]))
    num_seg = min(len(y) // LEN_SEG, num_seg) # 防止X比y长
    y = torch.tensor([y[LEN_SEG*i:LEN_SEG*i+LEN_SEG] for i in range(num_seg)], dtype=torch.float32)
    y = convert_y(y)

    return X[:num_seg], y[:num_seg]

def f02img(y):
    N = y.size(0)
    img = torch.zeros([N, 321, LEN_SEG], dtype=torch.float32)
    for i in range(N):
        img[i, y[i], torch.arange(LEN_SEG)] = 1
    return img

def pos_weight(data):
    N = data.size(0)
    non_melody = torch.sum(data[:, 0, :]).item() + 1
    melody = (N * LEN_SEG) - non_melody + 2
    z = torch.zeros((321, LEN_SEG), dtype=torch.float32)

    z[1:,:] += non_melody / melody
    z[0,:] += melody / non_melody
    return z

def ce_weight(data):
    N = data.size(0)
    non_melody = torch.sum(data == 0) + 1
    melody = (N * LEN_SEG) - non_melody + 2
    z = torch.zeros(321, dtype=torch.float32)
    z[1:] += non_melody / melody
    z[0] += melody / non_melody
    return z
