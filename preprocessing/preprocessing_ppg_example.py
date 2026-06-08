import subprocess
import pandas as pd
import numpy as np
import ast
import os
import tqdm
import shutil
import wfdb
from scipy import signal

'''
Function to create patches for NeuroRVQ
'''
def create_patches(ppg_signal, maximum_patches, patch_size, channels_use):
    n, c, t = ppg_signal.shape  # Batch / trials, channels, time
    n_time = (maximum_patches // len(channels_use))
    ppg_signal = ppg_signal[:, :, :n_time * patch_size]
    ppg_signal_patches = ppg_signal[:, channels_use, :]
    return ppg_signal_patches, n_time


# Get an out-of-distribution (dataset not used during training) example from https://physionet.org/content/bidmc/1.0.0/
record = wfdb.rdrecord(
    "bidmc03",
    pn_dir="bidmc/1.0.0"

)
# Get the PPG signal
ppg_signal = record.p_signal[:, record.sig_name.index('PLETH,')]
ppg_signal = ppg_signal.reshape(1, -1)

# Pre-process based on NeuroRVQ specs
highpass = 0.5
lowpass = 40
lowpass_applied = min(lowpass, record.fs / 2) - 0.5
[b, a] = signal.butter(N=3, Wn=[highpass, lowpass_applied], btype='bandpass', fs=record.fs)
ppg_signal = signal.filtfilt(b, a, ppg_signal, axis=-1)
# Resample to 100Hz
ppg_signal = signal.resample(ppg_signal, num=int(ppg_signal.shape[1] / record.fs * 100), axis=-1)
ppg_signal = ppg_signal.astype('float16')
ppg_signal = ppg_signal.reshape(1, 1, ppg_signal.shape[-1])
np.save("./example_files/ppg_sample/example_ppg.npy", ppg_signal)
