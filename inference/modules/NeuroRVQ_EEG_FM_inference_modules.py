import numpy as np
import torch

# List of all channels used in the pre-training of "NeuroRVQ_EEG_v1" model
ch_names_global = np.array([b'a1', b'a2', b'af3', b'af4', b'af7', b'af8', b'afz', b'c1', b'c2',
                            b'c3', b'c4', b'c5', b'c6', b'ccp1', b'ccp2', b'ccp3', b'ccp4',
                            b'ccp5', b'ccp6', b'ccp7', b'ccp8', b'cfc1', b'cfc2', b'cfc3',
                            b'cfc4', b'cfc5', b'cfc6', b'cfc7', b'cfc8', b'cp1', b'cp2',
                            b'cp3', b'cp4', b'cp5', b'cp6', b'cpz', b'cz', b'eog', b'f1',
                            b'f10', b'f2', b'f3', b'f4', b'f5', b'f6', b'f7', b'f8', b'f9',
                            b'fc1', b'fc2', b'fc3', b'fc4', b'fc5', b'fc6', b'fcz', b'fp1',
                            b'fp2', b'fpz', b'ft7', b'ft8', b'fz', b'iz', b'loc', b'o1', b'o2',
                            b'oz', b'p08', b'p1', b'p10', b'p2', b'p3', b'p4', b'p5', b'p6',
                            b'p7', b'p8', b'p9', b'po1', b'po10', b'po2', b'po3', b'po4',
                            b'po7', b'po8', b'po9', b'poz', b'pz', b'roc', b'sp1', b'sp2',
                            b't1', b't10', b't2', b't3', b't4', b't5', b't6', b't7', b't8',
                            b't9', b'tp10', b'tp7', b'tp8', b'tp9'])

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
