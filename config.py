"""
Ke Chen knutchen@ucsd.edu

Tone-Octave Network - config file

This file contains all constants, hyperparameters, and settings for the model

"""

exp_name = "SpecTNT"
# file path
model_type = "SpecTNT" # MCDNN, FTANet, MSNet, MLDRNet
data_path = "/home/la/kechen/Research/KE_SpecTNT/data"
train_file = "data/train_data.txt"
test_file = [
    "data/test_adc.txt",
    "data/test_mirex.txt",
    "data/test_melody.txt"
]
    
save_path = "/home/la/kechen/Research/KE_SpecTNT/model_backup"
resume_checkpoint = '/home/la/kechen/Research/KE_SpecTNT/TONet/model_backup/bestk_2.ckpt'
# "model_backup/TO-FTANet_adc_best.ckpt" # the model checkpoint

# train config
batch_size = 16
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
ablation_mode = "single" # single, tcfp, spl, spat, all
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

# spectnt config
mel_bins = 360 # cfp bins not mel bins
enable_tscam = True
htsat_attn_heatmap = False
loss_type = 'clip_bce'
spectnt_ckpt = None # "/home/la/kechen/Research/KE_SpecTNT/ckpt/swin_base_patch4_window12_384_22k.pth"