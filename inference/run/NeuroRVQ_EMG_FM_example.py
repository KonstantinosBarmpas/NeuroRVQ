import yaml
import numpy as np
import torch
from NeuroRVQ_EMG.NeuroRVQ import NeuroRVQFM
from NeuroRVQ_EMG.NeuroRVQ_modules import get_encoder_decoder_params
from inference.modules.NeuroRVQ_EMG_tokenizer_inference_modules import ch_names_global, create_embedding_ix, check_model_eval_mode
from functools import partial
from torch import nn

def load_neurorqv_fm(verbose=False,
                     model_path='./pretrained_models/foundation_models/NeuroRVQ_EMG_foundation_model_v1.pt'):
    # Load experiment parameters from config file
    config_stream = open("./flags/NeuroRVQ_EMG_v1.yml", 'r')
    args = yaml.safe_load(config_stream)

    # Fix the seeds for reproducibility
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get configuration params
    args['n_global_electrodes'] = len(ch_names_global)
    encoder_config, decoder_config = get_encoder_decoder_params(args)

    # Load the foundation model
    foundation_model = NeuroRVQFM(n_patches=args['n_patches'],
                                    patch_size=args['patch_size'],
                                    in_chans=args['in_chans_second_stage'],
                                    out_chans=args['out_chans_second_stage'],
                                    num_classes=0,
                                    embed_dim=args['embed_dim_second_stage'],
                                    depth=args['depth_second_stage'],
                                    num_heads=args['num_heads_second_stage'],
                                    mlp_ratio=args['mlp_ratio_second_stage'], qkv_bias=args['qkv_bias_second_stage'],
                                    qk_norm=partial(nn.LayerNorm, eps=1e-6), drop_rate=args['drop_rate_second_stage'],
                                    attn_drop_rate=args['attn_drop_rate_second_stage'],
                                    drop_path_rate=args['drop_path_rate_second_stage'],
                                    init_values=args['init_values_second_stage'],
                                    init_scale=args['init_scale_second_stage'],
                                    n_global_electrodes=args['n_global_electrodes'],
                                    use_as_encoder = True, vocab_size=args['n_code'],
                                    use_for_pretraining = False).to(device)

    missing_keys, unexpected_keys = foundation_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    total_params = sum(p.numel() for p in foundation_model.parameters())
    print(f"Total parameters: {total_params}")

    if (verbose):
        print(f"Missing keys: {missing_keys},\nUnexpected keys: {unexpected_keys}")
