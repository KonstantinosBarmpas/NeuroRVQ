import os
import csv
from fine_tuning.wrappers import NeuroRVQWrapper

class CSVLogger():
    def __init__(self, output_dir, ex_id):
        self.log_dir = os.path.join(output_dir, f"{ex_id}_log")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self._files = set()
    
    def report_scalar(self, title, series, value, iteration):
        '''
        Mimics clearml report_scalar() function to log values to CSV file
        '''
        if 'train' in series:
            filepath = os.path.join(self.log_dir, f"{title}_train.csv")
        else:
            filepath = os.path.join(self.log_dir, f"{title}_val.csv")

        write_header = filepath not in self._files

        with open(filepath, mode="a", newline="") as f:
            writer = csv.writer(f)
            if 'MEAN' in title:
                if write_header:
                    writer.writerow(["Series", "Iteration", "Value"])
                    self._files.add(filepath)
                writer.writerow([series, iteration, value])
            else:
                if write_header:
                    writer.writerow(["Fold", "Iteration", "Value"])
                    self._files.add(filepath)
                writer.writerow([series.split(' ')[-1], iteration, value])

def get_logger():
    logger = CSVLogger("results", 0)
    return logger

def get_model(ch_names, n_times, n_outputs, args, foundation_model, train_head_only=False):
    """
    Returns: FinetuningWrapper for the specified model
    """
    return NeuroRVQWrapper(
        n_time=n_times,
        ch_names=ch_names,
        n_outputs=n_outputs,
        train_head_only=train_head_only,
        args = args,
        foundation_model = foundation_model
    )
