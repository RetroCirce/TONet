"""
Ke Chen knutchen@ucsd.edu

Tone-Octave Network - config file

This file contains all constants, hyperparameters, and settings for the model

"""


exp_name = "FTANet"
# file path
model_type = "FTANet" # MCDNN, FTANet, MSNet, MLDRNet
data_path = "data"
train_file = "data/train_data.txt"
test_file = [
    "data/test_adc.txt",
    "data/test_mirex.txt",
    "data/test_melody.txt"
]
    
save_path = "model_backup"
resume_checkpoint = None 
# "model_backup/TO-FTANet_adc_best.ckpt" # the model checkpoint

# train config
batch_size = 12
lr = 1e-4
epochs = 1000
n_workers = 4
save_period = 1
tone_class = 12 # 60
octave_class = 8 # 6
random_seed = 19961206
max_epoch = 500
freq_bin = 360
'''
single: the original network
tcfp: with TCFP but without tone-octave fusion
spl: with tone-octave fustion in linear layer but without tcfp
spat: with tone-octave fusion in attention layer
all: the full tone-octave network
'''
ablation_mode = "all" # single, tcfp, spl, spat, all
startfreq = 32
stopfreq = 2050
cfp_dir = "cfp_360_new"

# feature config
fs = 8000.0
hop = 80.0
octave_res = 60
seg_dur = 1.28 # sec
seg_frame = int(seg_dur * fs // hop)
shift_dur = 1.28 # sec
shift_frame = int(shift_dur * fs // hop)
