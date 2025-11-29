import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Define EEG bands
bands = {
    "Delta (0.5–4 Hz)": (0.5, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–13 Hz)": (8, 13),
    "Beta (13–30 Hz)": (13, 30),
    "Gamma (30–45 Hz)": (30, 45),
}

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def plot_reconstructions(originals_list, reconstructions_list, fs,
                         labels=["NeuroRVQ"], save_dir="./figures"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    N, T = originals_list[0].shape
    time = np.linspace(0, T / fs, T)

    for i in tqdm(range(N), desc="Samples"):
        plt.figure(figsize=(10, 12))

        # Plot raw signals
        plt.subplot(6, 1, 1)
        orig = originals_list[0][i]
        recon = reconstructions_list[0][i]
        label = labels[0]

        plt.plot(time, orig, label=f"Original Signal", alpha=0.7)
        plt.plot(time, recon, linestyle='--', label=f"{label} Reconstruction", alpha=0.7)

        plt.title(f"Raw Signal")
        plt.legend()
        plt.ylabel("Amplitude")

        # Plot filtered bands
        for j, (band_name, (low, high)) in enumerate(bands.items()):
            plt.subplot(6, 1, j + 2)
            orig = originals_list[0][i]
            recon = reconstructions_list[0][i]
            label = labels[0]

            orig_band = bandpass_filter(orig, low, high, fs)
            recon_band = bandpass_filter(recon, low, high, fs)

            plt.plot(time, orig_band, label=f"{label} Original Signal", alpha=0.7)
            plt.plot(time, recon_band, linestyle='--', label=f"{label} Reconstruction", alpha=0.7)

            plt.title(f"{band_name} Band")
            plt.ylabel("Amplitude")

        plt.xlabel("Time (s)")
        plt.tight_layout()

        plt.savefig(f"{save_dir}/sample_{i}.png")
        plt.close()


def process_and_plot(originals, reconstructions, fs):
    P, T = reconstructions[0].shape

    originals_np = [
        original.detach().cpu().numpy().reshape(P, T)
        for original in originals
    ]
    reconstructions_np = [
        reconstruction.detach().cpu().numpy().reshape(P, T)
        for reconstruction in reconstructions
    ]

    # Plot
    plot_reconstructions(originals_np, reconstructions_np, fs)
