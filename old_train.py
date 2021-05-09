import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import numpy as np

from old_util import *
from model.msnet import MSnet


import config
import time

import argparse

class Dataset(Data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def train(train, test, epoch_num, batch_size, lr, gid, op, pretrained=None):

    torch.backends.cudnn.enabled = False

    Net = MSnet()
    if pretrained is not None:
        Net.load_state_dict(torch.load(pretrained))
    
    if gid is not None:
        Net.cuda()
    else:
        Net.cpu()
    Net.float()

    epoch_num = epoch_num
    batch_size = batch_size
    lr = lr
    
    X_train, y_train = load_train_data(path=train)
    test_list = load_list(path=test, mode='test')

    pw = pos_weight(f02img(y_train))
    if gid is not None:
        pw = pw.cuda()

    data_set = Dataset(data_tensor=X_train, target_tensor=f02img(y_train))
    data_loader = Data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)

    best_epoch = 0
    best_OA = 0
    time_series = np.arange(64) * 0.01

    BCELoss = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.Adam(Net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, 200)
    tick = time.time()
    for epoch in range(epoch_num):
        tick_e = time.time()
        Net.train()
        train_loss = 0

        for step, (batch_x, batch_y) in enumerate(data_loader):

            opt.zero_grad()
            if gid is not None:
                pred, _ = Net(batch_x.cuda())
                pred = pred[:,0]
                loss = BCELoss(pred, batch_y.cuda())
            else:
                pred = Net(batch_x)
                pred = pred[:,0]
                loss = BCELoss(pred, batch_y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            
        
        Net.eval()
        eval_arr = np.zeros(5, dtype=np.double)
        with torch.no_grad():
            for i in range(len(test_list)):
                X_test, y_test = load_data(test_list[i])
                if gid is not None:
                    pred, _ = Net(X_test.cuda())
                    pred = pred[:,0]
                else:
                    pred, _  = Net(X_test)
                    pred = pred[:,0]
                est_freq = pred2res(pred.cpu()).flatten()
                ref_freq = y2res(y_test).flatten()
                time_series = np.arange(len(ref_freq)) * 0.01
                eval_arr += melody_eval(time_series, ref_freq, time_series, est_freq)
                
            eval_arr /= len(test_list)
            train_loss /= step + 1

        scheduler.step()

        print("----------------------")
        print("Epoch={:3d}\tTrain_loss={:6.4f}\tLearning_rate={:6.4f}e-4".format(epoch, train_loss, 1e4 * opt.state_dict()['param_groups'][0]['lr']))
        print("Valid: VR={:.2f}\tVFA={:.2f}\tRPA={:.2f}\tRCA={:.2f}\tOA={:.2f}".format(eval_arr[0], eval_arr[1], eval_arr[2], eval_arr[3], eval_arr[4]))
        if eval_arr[-1] > best_OA:
            best_OA = eval_arr[-1]
            best_epoch = epoch
        # torch.save(Net.state_dict(), op+'{:.2f}_{:d}'.format(eval_arr[4], epoch))
        print('Best Epoch: ', best_epoch, ' Best OA: ', best_OA)
        print("Time: {:5.2f}(Total: {:5.2f})".format(time.time() - tick_e, time.time() - tick))


def parser():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('-train', '--train_list_path', 
                    help='the path of training data list (default: %(default)s)',
                    type=str, default='./train_npy.txt')
    p.add_argument('-test', '--test_list_path', 
                    help='the path of test data list (default: %(default)s)',
                    type=str, default='./test_04_npy.txt')
    p.add_argument('-ep', '--epoch_num', 
                    help='the number of epoch (default: %(default)s)',
                    type=int, default=100)
    p.add_argument('-bs', '--batch_size', 
                    help='The number of batch size (default: %(default)s)',
                    type=int, default=50)
    p.add_argument('-lr', '--learning_rate', 
                    help='the number of learn rate (default: %(default)s)',
                    type=float, default=0.0001)    
    p.add_argument('-gpu', '--gpu_index',
                    help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s)',
                    type=int, default=0)
    p.add_argument('-o', '--output_dir',
                    help='Path to output folder (default: %(default)s)',
                    type=str, default='./model')
    p.add_argument('-pm', '--pretrained_model',
                    help='the path of pretrained model (Transformer or Streamline) (default: %(default)s)',
                    type=str, default=None)
    
    return p.parse_args()

if __name__ == '__main__':
    train(
        train = config.train_file,
        test = config.test_file[0],
        epoch_num = 100,
        batch_size = 50,
        lr = 1e-4,
        gid = 0,
        op = None,
        pretrained=None
    )


    # args = parser()
    # if args.gpu_index is not None:
    #     with torch.cuda.device(args.gpu_index):
    #         train(args.train_list_path, args.test_list_path, args.epoch_num, args.batch_size, args.learning_rate, args.gpu_index, args.output_dir, args.pretrained_model)
    # else: 
    #     train(args.train_list_path, args.test_list_path, args.epoch_num, args.batch_size, args.learning_rate, args.gpu_index, args.output_dir, args.pretrained_model)
