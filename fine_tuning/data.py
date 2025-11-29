import numpy as np
import pandas as pd
from abc import ABC

class Benchmark(ABC):
    """
    Class for benchmark dataset with expected properties:
        eeg: array of EEG data (samples, channels, time)
        subject_ids: array of subject ID for each data sample (samples,)
        labels: array of target class labels for each data sample (samples,)
        chnames: array of electrode channel names (channels,)
    """
    def __init__(self):
        self.eeg = None
        self.subject_ids = None
        self.labels = None
        self.chnames = None

    def get_data(self):
        return self.eeg, self.subject_ids, self.labels, self.chnames

    def sample_balanced_set(self, idx, seed):
        """
        Performs a random sampling of indices to balance classes for each subject
            idx: array of sample indices relative to self.eeg
            seed: random seed for sampling
        Returns:
            filtered indices after random sampling
        """
        rng = np.random.default_rng(seed)
        subj_all = self.subject_ids[idx]
        y_all = self.labels[idx]
        sampled = []

        for s in np.unique(subj_all):
            mask_s = (subj_all == s)
            idx_s = idx[mask_s]
            y_s = y_all[mask_s]

            labels = np.unique(y_s)

            idx_by_label = [idx_s[y_s == label] for label in labels]

            # minority per subject
            n = min([len(idx_l) for idx_l in idx_by_label])
            if n == 0:
                continue

            take_by_label = [rng.choice(idx_l, size=n, replace=False) for idx_l in idx_by_label]
            sampled.append(np.concatenate(take_by_label))

        sampled_idx = np.concatenate(sampled)
        return sampled_idx

class YourCustomBenchmark(Benchmark):
    """
    Custom Class Example where your eeg trials are in stored in .npy file
    The labels and other info in the .pd file
    And your dasaset has 4-classes
    """
    def __init__(self, root, subdir, apply_car):
        super().__init__()
        print("Loading Your Data...")
        eeg = np.load('./fine_tuning/data/data_eeg.npy', mmap_mode='r')
        tf = pd.read_pickle('./fine_tuning/data/trial_features.pd')
        subject_ids = tf['subject_id'].to_numpy()
        chnames = np.array([c.upper() for c in tf.attrs['channel_names']])
        labels = tf['task'].replace({'class_1': 0, 'class_2': 1, 'class_3': 2, 'class_4': 3}).to_numpy()

        self.eeg = eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames

    def sample_balanced_set(self, idx, seed):
        print("Classes are already balanced for High Gamma")
        return idx

def load_benchmark(benchmark, root, subdir, apply_car=False) -> Benchmark:
    BENCHMARK_CLASSES = {
        "Custom Benchmark": YourCustomBenchmark
    }

    assert (benchmark in BENCHMARK_CLASSES), f"Unsupported benchmark {benchmark}. Make sure load function is added to BENCHMARK_LOADERS."

    benchmark_cls = BENCHMARK_CLASSES[benchmark]
    return benchmark_cls(root, subdir, apply_car)
