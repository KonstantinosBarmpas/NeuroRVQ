import pyxdf
from scipy import signal
import numpy as np
import os
import urllib.request
import tarfile
import h5py

def fix_length(emg_signal, target_length=120_000):
    """
    Ensures a 2D EMG signal (time x channels) is exactly target_length samples long.
    Pads with zeros or trims only along the time axis.
    """
    current_length = emg_signal.shape[0]

    if current_length > target_length:
        # Trim along time dimension
        return emg_signal[:target_length, :]
    elif current_length < target_length:
        # Pad with zeros at the end along the time axis
        pad_amount = target_length - current_length
        padding = ((0, pad_amount), (0, 0))  # pad only time axis
        return np.pad(emg_signal, padding, mode='constant')
    else:
        # Already correct length
        return emg_signal
        
'''
Example of how to preprocess an EMG recording from emg2pose
High Pass: 400Hz
Low Pass: 20Hz
Resample at: 1000Hz
'''
def preprocessing_emg():

    # Download file
    print("Downloading emg2pose (mini)...")
    url = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar"
    tar_path = os.path.join('./example_files/emg_sample', "emg2pose_dataset_mini.tar")
    urllib.request.urlretrieve(url, tar_path)
    # Extract tar file
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path='./example_files/emg_sample')

    print("Download and extraction complete.")

    target_fs = 1000
    highpass = 20
    lowpass = 400
    
    # Loop over the downloaded hdf5 files
    f_names = [d for d in os.listdir('./example_files/emg_sample/emg2pose_dataset_mini') if d.endswith('hdf5')]
    
    # Get sample rate from documentation
    fs = 2000
    
    # Hardcode the channel names
    ch_names = np.array(['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16'])
    
    for f_i in f_names:
        
        with h5py.File(os.path.join('./example_files/emg_sample/emg2pose_dataset_mini', f_i), 'r') as f_emg:
            dset = f_emg['emg2pose']['timeseries']
            # Load everything into memory at once (structured NumPy array)
            data = dset[:]
            
        x = fix_length(data['emg']).T
    
        lowpass_applied = min(lowpass, fs / 2) - 0.5
        [b, a] = signal.butter(N=3, Wn=[highpass, lowpass_applied], btype='bandpass', fs=fs)
        x = signal.filtfilt(b, a, x, axis=-1)
        
        # Resampling
        if target_fs != fs:
            x = signal.resample(x, num=int(x.shape[-1] / fs * target_fs), axis=-1)
        
        # Convert to float16 only after filtering
        x = x.astype('float16')
        x = x.reshape(1, x.shape[0], x.shape[1])
        ch_names = np.array([c.lower().encode() for c in ch_names])
        break
        
    return x, ch_names

'''
Function to create patches for NeuroRVQ
'''
def create_patches(emg_signal, maximum_patches, patch_size, channels_use):
    n, c, t = emg_signal.shape  # Batch / trials, channels, time
    n_time = (maximum_patches // len(channels_use))
    emg_signal = emg_signal[:, :, :n_time * patch_size]
    emg_signal_patches = emg_signal[:, channels_use, :]
    return emg_signal_patches, n_time
