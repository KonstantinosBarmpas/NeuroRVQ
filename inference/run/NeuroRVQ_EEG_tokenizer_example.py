import yaml
import numpy as np
import torch
from NeuroRVQ.NeuroRVQ import NeuroRVQTokenizer
from NeuroRVQ.NeuroRVQ_modules import get_encoder_decoder_params
from preprocessing.preprocessing_eeg_example import preprocessing_eeg
from plotting.plotting_example import process_and_plot
from preprocessing.preprocessing_eeg_example import create_patches
from inference.modules.NeuroRVQ_EEG_tokenizer_inference_modules import ch_names_global, create_embedding_ix, check_model_eval_mode

def load_neurorqv_tokenizer(run_example=False, plot_results=False, verbose=False,
                            model_path='./pretrained_models/tokenizers/NeuroRVQ_EEG_tokenizer_v1.pt'):
    # Load experiment parameters from config file
    config_stream = open("./flags/NeuroRVQ_EEG_v1.yml", 'r')
    args = yaml.safe_load(config_stream)

    # Fix the seeds for reproducibility
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get configuration params for tokenizer
    args['n_global_electrodes'] = len(ch_names_global)
    encoder_config, decoder_config = get_encoder_decoder_params(args)

    # Load the tokenizer
    tokenizer = NeuroRVQTokenizer(encoder_config, decoder_config, n_code=args['n_code'],
                                  code_dim=args['code_dim'], decoder_out_dim=args['decoder_out_dim']).to(device)

    tokenizer.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    tokenizer.eval()

    if (verbose):
        check_model_eval_mode(tokenizer)

    if (run_example):
        x, ch_names = preprocessing_eeg('example_files/eeg_sample/example_eeg_file.xdf')
        ch_mask = np.isin(ch_names, ch_names_global)
        ch_names = ch_names[ch_mask]
        x = x[:, ch_mask, :]

        x, n_time = create_patches(x, maximum_patches=args['n_patches'], patch_size=args['patch_size'], channels_use=ch_mask)
        x = torch.from_numpy(x).float().to(device)

        temporal_embedding_ix, spatial_embedding_ix = create_embedding_ix(n_time, args['n_patches'], ch_names, ch_names_global)
        oringal_signal_std, reconstructed_signal_std = tokenizer(x, temporal_embedding_ix.int().to(device), spatial_embedding_ix.int().to(device))

        if (plot_results):
            process_and_plot(oringal_signal_std, reconstructed_signal_std, fs=args['patch_size'])
