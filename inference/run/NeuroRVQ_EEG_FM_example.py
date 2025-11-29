import yaml
import numpy as np
import torch
from NeuroRVQ.NeuroRVQ import NeuroRVQFM
from NeuroRVQ.NeuroRVQ_modules import get_encoder_decoder_params
from inference.modules.NeuroRVQ_EEG_tokenizer_inference_modules import ch_names_global, create_embedding_ix, check_model_eval_mode
from functools import partial
from torch import nn
from fine_tuning.utils import get_logger, get_model
from fine_tuning.data import load_benchmark
import skorch

def perform_finetuning(benchmarks, metrics, args, foundation_model):
    '''
    Performs full finetuning on benchmarks using all data for training (no folds, no validation set)
    Saves finetuned model, no metrics returned
    '''
    logger = get_logger()
    results = {}
    for benchmark in benchmarks:
        # Load data stored in "./fine_tuning/data" folder - see example in data.py
        b = load_benchmark(benchmark,  './', "NeuroRVQ")
        X, sbj_id, y, ch_names = b.get_data()
        n_outputs = len(np.unique(y))
        n, c, t = X.shape
        dataset = skorch.dataset.Dataset(X[:-1], y[:-1])
        dummy_val = skorch.dataset.Dataset(np.array([X[0]]), np.array([y[0]]))

        # Make model
        model = get_model(
            ch_names=ch_names,
            n_times=t,
            n_outputs=n_outputs,
            args = args,
            foundation_model = foundation_model,
            train_head_only=args['train_head_only_finetuning']
        )
        print(f"No. Trainable Parameters: {model.size()}")

        # Finetune model
        model.fit(
            dataset,
            dummy_val,
            batch_size=args['batch_size_finetuning'],
            epochs=args['epoch_finetuning']
        )

        # Log training results (per epoch)
        for m in metrics:
            results = model.results[f'train_{m}']
            for i in range(args['epoch_finetuning']):
                logger.report_scalar(title="Fine-Tuning NeuroRVQ", series=f'train',
                                     value=results[i], iteration=i)

        # Save model
        torch.save(model.state_dict(), './fine_tuned_model.pt')

    return

def load_neurorqv_fm(fine_tuning=False, verbose=False,
                     model_path='./pretrained_models/foundation_models/NeuroRVQ_EEG_foundation_model_v1.pt'):
    # Load experiment parameters from config file
    config_stream = open("./flags/NeuroRVQ_EEG_v1.yml", 'r')
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

    if (fine_tuning):
        # Select benchmark datasets
        benchmarks = ["High Gamma"]
        # Select evaluation metrics
        # NOTE: metrics not included in this list will need to be implemented in the module for each model
        metrics = ["accuracy", "bacc"]
        perform_finetuning(benchmarks, metrics, args, foundation_model)
