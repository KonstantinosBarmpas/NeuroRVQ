import subprocess
import pandas as pd
import numpy as np
import ast
import os
import tqdm
import shutil
import wfdb


##############################
# Class Aggregation Function #
##############################
def aggregate_diagnostic_one_class(y_dic):
    best_class = "None"
    best_value = -float('inf')
    for key, value in y_dic.items():
        if key in agg_df.index and value > best_value:
            best_value = value
            best_class = agg_df.loc[key].diagnostic_class
    return best_class

def aggregate_diagnostic_one_subclass(y_dic):
    best_class = "None"
    best_value = -float('inf')
    for key, value in y_dic.items():
        if key in agg_df.index and value > best_value:
            best_value = value
            best_class = agg_df.loc[key].diagnostic_subclass
    return best_class

def aggregate_diagnostic_all_classes(y_dic):
    best_class = "None"
    best_value = -float('inf')
    for key, value in y_dic.items():
        if key in agg_df.index and value > best_value:
            best_value = value
            best_class = key
    return best_class
    
'''
Function to create patches for NeuroRVQ
'''
def create_patches(ecg_signal, maximum_patches, patch_size, channels_use):
    n, c, t = ecg_signal.shape  # Batch / trials, channels, time
    n_time = (maximum_patches // len(channels_use))
    ecg_signal = ecg_signal[:, :, :n_time * patch_size]
    ecg_signal_patches = ecg_signal[:, channels_use, :]
    return ecg_signal_patches, n_time


if not os.path.exists("./example_files/ecg_sample/ptb_xl_cut_benchmarking.npy"):
    ##############################
    #      Downloading PTB-XL    #
    ##############################
    print("Downloading ptb_xl....")
    subprocess.run([
        "wget",
        "-r", "-N", "-c", "-np",
        "-P", "./ptb_xl",
        "https://physionet.org/files/ptb-xl/1.0.3/"
    ], check=False)


    ##############################
    #     Merging Folders        #
    ##############################
    source_root = "./ptb_xl/physionet.org/files/ptb-xl/1.0.3/records500"
    target_root = "./ptb_xl/records500_all"
    os.makedirs(target_root, exist_ok=True)

    for subfolder in os.listdir(source_root):
        subfolder_path = os.path.join(source_root, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                src_file = os.path.join(subfolder_path, filename)
                dst_file = os.path.join(target_root, filename)

                # If filename already exists, rename to avoid overwrite
                if os.path.exists(dst_file):
                    name, ext = os.path.splitext(filename)
                    dst_file = os.path.join(target_root, f"{name}_{subfolder}{ext}")

                shutil.move(src_file, dst_file)

    print("Merge complete.")
    print("Dataset cutting....")

    ##############################
    #     Dataset Cutting        #
    ##############################
    path = './ptb_xl/physionet.org/files/ptb-xl/1.0.3/'
    data_path = './ptb_xl/records500_all'

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    new_Y = Y.loc[:, ['patient_id', 'scp_codes', 'strat_fold', 'filename_lr', 'filename_hr']]
    new_Y.index.name == 'ecg_id'

    # Load scp_statements.csv for diagnostic aggregation
    agg_df_all = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df_all[agg_df_all.diagnostic == 1]

    # Apply diagnostic superclass
    new_Y['diagnostic_5_classes'] = new_Y.scp_codes.apply(aggregate_diagnostic_one_class)
    new_Y['diagnostic_23_classes'] = new_Y.scp_codes.apply(aggregate_diagnostic_one_subclass)
    new_Y['diagnostic_44_classes'] = new_Y.scp_codes.apply(aggregate_diagnostic_all_classes)
    new_Y['filename_hr'] = new_Y['filename_hr'].str.split('/').str[-1]
    new_Y.to_csv('./example_files/ecg_sample/ptb_xl_cut_benchmarking.csv')

    # Initialize 3D array: files x channels x time
    X = np.zeros((len(new_Y['filename_hr']), 12, 5000))

    for idx, f_i in enumerate(tqdm.tqdm(new_Y['filename_hr'])):
        x = wfdb.rdsamp(os.path.join(data_path, f_i))[0].T
        if x.shape != (12, 5000):
            raise ValueError(f"Signal {f_i} has shape {x.shape}, expected (12, 5000)")
        ch_names = np.array(wfdb.rdsamp(os.path.join(data_path, f_i.split('.')[0]))[1]['sig_name'])
        ch_names = np.array([e.lower() for e in ch_names])
        if (idx==0):
            ref_ch_names = ch_names
            X[idx, :, :] = x
        else:
            try:
                # Find the indices that map current channels to reference order
                reorder_idx = [np.where(ch_names == ch)[0][0] for ch in ref_ch_names]
                x_reor = x[reorder_idx, :]
            except IndexError:
                raise ValueError(f"Channel names in {f_i} do not match reference channels.")
            X[idx, :, :] = x_reor

    print(X.shape)
    np.save('./example_files/ecg_sample/ptb_xl_cut_benchmarking.npy', X)
    print("Dataset is ready")
