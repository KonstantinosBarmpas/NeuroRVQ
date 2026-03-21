import numpy as np
import torch

# List of all channels used in the pre-training of "NeuroRVQ_ECG_v1" model
ch_names_global = np.array([b'avf', b'avl', b'avr', b'i', b'ii', b'iii', b'v1', b'v2', b'v3', b'v4', b'v5', b'v6', b'vx', b'vy', b'vz'])

def check_model_eval_mode(model):
    for name, module in model.named_modules():
        if hasattr(module, 'training'):
            if module.training:
                print(f"[WARNING] Module {name} is still in training mode.")
            else:
                print(f"[OK] Module {name} is in eval mode.")

def create_embedding_ix(n_time, max_n_patches, ch_names_sample, ch_names_global):
    """Creates temporal and spatial embedding indices for a sample with given regular shape.
    Args:
        n_time: Int. Number of patches along the time dimension
        max_n_patches: The maximum number of patches, for aligning the current time-point to the right.
        ch_names_sample (n_channels_sample,): The specific channel names of the sample
        ch_names_global (n_channels_global): The reference channel names of the model
    Returns:
        temp_embed_ix (1, n_patches): tensor
        spat_embed_ix (1, n_patches): tensor
    """
    # Temporal embedding ix
    temp_embed_ix = torch.arange(max_n_patches - n_time, max_n_patches)
    temp_embed_ix = temp_embed_ix.repeat(len(ch_names_sample))
    temp_embed_ix = temp_embed_ix.reshape(1, -1)

    # Spatial embedding ix
    spat_embed_ix = torch.tensor([np.where(ch_names_global == c)[0][0] for c in ch_names_sample])
    spat_embed_ix = torch.repeat_interleave(spat_embed_ix, n_time)
    spat_embed_ix = spat_embed_ix.reshape(1, -1)

    return temp_embed_ix, spat_embed_ix
