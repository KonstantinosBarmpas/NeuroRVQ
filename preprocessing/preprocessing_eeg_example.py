import pyxdf
from scipy import signal
import numpy as np

'''
Example of how to preprocess an EEG recording
Notch filter: 50., 60., 100.Hz
Band-Pass: 0.5-45 Hz (3rd-order Butterworth; upper edge applied at 44.5 Hz)
Clip: +/-500
Resample at: 200Hz
'''
def preprocessing_eeg(data_path):
    notch = [50., 60., 100.]
    target_fs = 200
    highpass = 0.5
    lowpass = 45
    clip = 500
    streams, header = pyxdf.load_xdf(data_path, verbose=False,
                                     synchronize_clocks=True, dejitter_timestamps=True,
                                     select_streams=[{'name': 'Quick-32r_R2_EEG'}])
    # Get sample rate
    fs = float(streams[0]['info']['nominal_srate'][0])
    channel_information = streams[0]["info"]["desc"][0]["channels"][0]["channel"]
    ch_names = [x["label"][0] for x in channel_information][:29]

    # Get EEG
    x = streams[0]["time_series"][:, :29].T.astype(np.float64)  # (channels, time) comes as float32

    # Filter and clip
    for f_notch in notch:
        if fs / 2 > f_notch:
            [b_notch, a_notch] = signal.iirnotch(w0=f_notch, Q=f_notch / 2, fs=fs)
            x = signal.filtfilt(b_notch, a_notch, x, axis=-1)
    lowpass_applied = min(lowpass, fs / 2) - 0.5
    [b, a] = signal.butter(N=3, Wn=[highpass, lowpass_applied], btype='bandpass', fs=fs)
    x = signal.filtfilt(b, a, x, axis=-1)
    x = x.clip(min=-clip, max=clip)
    # Resampling
    if target_fs != fs:
        x = signal.resample(x, num=int(x.shape[-1] / fs * target_fs), axis=-1)
    # Convert to float16 only after filtering
    x = x.astype('float16')
    x = x.reshape(1, x.shape[0], x.shape[1])
    ch_names = np.array([c.lower().encode() for c in ch_names])
    return x, ch_names

'''
Function to create patches for NeuroRVQ
'''
def create_patches(eeg_signal, maximum_patches, patch_size, channels_use):
    n, c, t = eeg_signal.shape  # Batch / trials, channels, time
    channels_use = np.asarray(channels_use, dtype=bool)
    if channels_use.shape[0] != c:
        raise ValueError(f"channels_use has length {channels_use.shape[0]}, but signal has {c} channels")

    n_keep = int(channels_use.sum())
    if n_keep == 0:
        raise ValueError("None of the channels match the model's vocabulary channel list")

    # Patches per channel: limited by the model's max input tokens and by the recording length
    n_time = min(maximum_patches // n_keep, t // patch_size)
    if n_time == 0:
        raise ValueError(f"Signal too short: need at least {patch_size} samples, got {t}")

    eeg_signal_patches = eeg_signal[:, channels_use, :n_time * patch_size]
    return eeg_signal_patches, n_time
