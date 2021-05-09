"""
Tone-Octave Network - config file

This file contains all constants, hyperparameters, and settings for the model

"""


exp_name = "MSNet"
# file path
data_path = "/home/kechen/Research/ISMIR-2021-TONET/data"
train_file = "/home/kechen/Research/ISMIR-2021-TONET/data/train_data.txt"
test_file = [
    "/home/kechen/Research/ISMIR-2021-TONET/data/test_adc.txt",
    "/home/kechen/Research/ISMIR-2021-TONET/data/test_mirex.txt",
    "/home/kechen/Research/ISMIR-2021-TONET/data/test_melody.txt"
]
    
save_path = "/data/home/knutchen/melody_ext/model_backup"


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
