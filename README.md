<div align="center">

<img src="images/banner.png" width="600">

# NeuroRVQ: Multi-Scale Biosignal Tokenization for Generative Foundation Models

<a href='https://neurips.cc/virtual/2026/poster/XXXXX'><img src='https://img.shields.io/badge/NeurIPS-2026-blue'></a>
<a href='https://arxiv.org/abs/2510.13068v4'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/ntinosbarmpas/NeuroRVQ'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>

[Konstantinos Barmpas](https://www.barmpas.com)<sup>1,2</sup> &emsp; [Na Lee](https://www.linkedin.com/in/na-lee-57777387/)<sup>1,2</sup> &emsp; [Dimitrios Chalatsis](http://dhalatsis.github.io)<sup>1</sup> &emsp; [William Raftery](https://www.linkedin.com/in/will-raf/)<sup>1</sup> &emsp;


[Yannis Panagakis](http://users.uoa.gr/~yannisp/)<sup>2,3,4</sup> &emsp; [Dimitrios Adamos](https://profiles.imperial.ac.uk/d.adamos)<sup>1,2</sup> &emsp; [Nikolaos Laskaris](https://people.auth.gr/laskaris/?lang=en)<sup>2,5</sup>  &emsp; 

[Alexandros Koliousis](https://akoliousis.com)<sup>6</sup> &emsp; [Dario Farina](https://profiles.imperial.ac.uk/d.farina)<sup>1</sup> &emsp;
[Stefanos Zafeiriou](https://profiles.imperial.ac.uk/s.zafeiriou)<sup>1,2</sup>  

<sup>1</sup>Imperial College London, United Kingdom <br>
<sup>2</sup>Cogitat, United Kingdom <br>
<sup>3</sup>National and Kapodistrian University of Athens, Greece <br>
<sup>4</sup>Archimedes Research Unit, Greece <br>
<sup>5</sup>Aristotle University of Thessaloniki, Greece <br>
<sup>6</sup>Northeastern University London, United Kingdom 

This is the official implementation of **NeuroRVQ**, a foundation model for biosignals powered by a state-of-the-art biosignal tokenizer. 

Visit the official website for more information: https://neurorvq.github.io

</div>

## 🌟 Overview

Biosignals such as electroencephalography (EEG), electrocardiography (ECG), and electromyography (EMG) encode physiological activity across multiple temporal and spectral scales, yielding representations that are rich but challenging for machine learning. Foundation models trained to predict masked signal tokens have shown promise in learning generalizable biosignal representations, yet their performance depends on the tokenizer's ability to preserve high-frequency dynamics and reconstruct signals with high fidelity. We introduce NeuroRVQ, a modality-adaptive biosignal tokenizer family designed for high-fidelity signal reconstruction. To capture the full frequency spectrum, NeuroRVQ decomposes biosignals into frequency-specific representations via multi-scale temporal convolutions, each encoded into hierarchical RVQ codebooks to preserve high-frequency detail, combined with a novel phase-aware training loss that respects the circular topology of Fourier phase. By tuning the temporal resolution, number and size of temporal kernels and RVQ depth, this design adapts to the spectro-temporal characteristics of each biosignal modality. To validate that tokenizer quality drives downstream performance, we train a simple masked-token foundation model for each modality (NeuroRVQ-FM) using the corresponding NeuroRVQ tokenizer. The NeuroRVQ-FM family achieves competitive or superior downstream performance compared to existing modality-specific foundation models, demonstrating that high-fidelity tokenization is a critical factor for effective biosignal modeling.

**NeuroRVQ Tokenizer** converts raw biosignals into compact and informative neural tokens. The input multi-variate time series is segmented into patches, encoded by the multi-scale temporal encoder at multiple resolutions, combined via a transformer encoder, then discretized into neural tokens through per-scale RVQ codebooks. Tokens are decoded to reconstruct the input patches using the Fourier spectrum.

**NeuroRVQ Foundation Model**  operates on the tokenized representation, using masked-token prediction with symmetric masking. By working at the token level, it captures long-range dependencies, learns abstract neural dynamics, and enables efficient pre-training across diverse biosignal datasets. The learned codebooks serve as prediction targets during pre-training, and the resulting representations transfer effectively to a range of downstream BCI tasks.

## Model and Modalities

| Model Version | Modality | Trained Models <a href='https://huggingface.co/ntinosbarmpas/NeuroRVQ'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a> | 
| :--- | :--- | :--- |
| **NeuroRVQ-EEG-tokenizer-v1** | EEG | NeuroRVQ_EEG_tokenizer_v1.pt |
| **NeuroRVQ-EEG-foundation-model-v1** | EEG | NeuroRVQ_EEG_foundation_model_v1.pt |
| **NeuroRVQ-EMG-tokenizer-v1** | EMG | NeuroRVQ_EMG_tokenizer_v1.pt |
| **NeuroRVQ-EMG-foundation-model-v1** | EMG | NeuroRVQ_EMG_foundation_model_v1.pt |
| **NeuroRVQ-ECG-tokenizer-v1** | ECG | NeuroRVQ_ECG_tokenizer_v1.pt |
| **NeuroRVQ-ECG-foundation-model-v1** | ECG | NeuroRVQ_ECG_foundation_model_v1.pt |
| **NeuroRVQ-PPG-tokenizer-v1** | PPG | NeuroRVQ_PPG_tokenizer_v1.pt |

## Tokenization / Reconstruction Capabilities

| EEG | ECG | EMG | PPG |
|:---:|:---:|:---:|:---:|
| <img src="images/eeg.png" width="300"/> | <img src="images/ecg.png" width="300"/> | <img src="images/emg.png" width="300"/> | <img src="images/ppg.png" width="300"/> |

## Downstream Performance

### EEG — Five Classification Tasks

| Model (Backbone Params) | Motor | ERP | Memory | Sleep | Eyes |
|---|---|---|---|---|---|
| BIOT (3.2M) | 44.3±7.9 | 50.0±0.0 | 51.0±1.8 | – | 76.3±4.9 |
| MIRepNet (5.2M) | 68.9±8.6 | – | – | – | – |
| CBraMod (4.9M) | 61.4±10.4 | 77.7±5.2 | 57.4±3.8 | 63.5±4.1 | 83.9±4.1 |
| EEGPT (25.7M) | 31.3±3.5 | 66.8±14.6 | 52.0±1.7 | 63.4±4.4 | 79.7±3.7 |
| NeuroGPT (79.5M) | 68.2±8.3 | 75.7±4.8 | **59.7±2.9** | <u>67.4±3.3</u> | 82.7±3.6 |
| CSBrain (4.9M) | 67.1±11.1 | 76.5±5.7 | 57.0±3.5 | 67.2±3.8 | 81.9±4.0 |
| REVE (69.2M) | <u>69.4±10.5</u> | <u>82.6±4.3</u> | <u>59.3±3.2</u> | 62.9±3.5 | 79.9±3.4 |
| BrainOmni (37.8M) | 58.5±5.8 | 72.3±4.1 | 51.8±2.3 | 57.6±5.5 | <u>85.2±2.2</u> |
| LaBraM (5.8M) | 63.0±7.6 | 82.2±4.0 | 52.6±2.6 | 65.2±3.7 | 79.9±4.7 |
| **NeuroRVQ-EEG†** (5.9M) | **70.0±7.3** | **87.6±3.3** | 57.4±2.7 | **72.8±2.8** | **86.9±2.6** |

We used the benchmark presented in IEEE MLSP 2025 Paper [Assessing the Capabilities of Large Brainwave Foundation Models](https://ieeexplore.ieee.org/document/11204282).

[Open-Source Benchmark Code Available](https://github.com/dykestra/EEG-Benchmarking)

### ECG — PTB-XL Classification

| Model (Backbone Params) | 5-class Accuracy | 5-class BAcc | 43-class Accuracy | 43-class BAcc |
|---|---|---|---|---|
| HuBERT-ECG (30.2M) | <u>72.60</u> | 60.23 | 62.49 | 20.71 |
| ECGFounder (30.7M) | **76.55** | **65.39** | <u>65.51</u> | <u>28.96</u> |
| **NeuroRVQ-ECG** (264K) | 70.19 | <u>64.50</u> | **79.17** | **58.33** |

[Open-Source Benchmark Code Available](https://github.com/dykestra/Biosignal-Benchmarking)

### EMG — Four Classification Tasks

| Model (Backbone Params) | Discrete Gestures BAcc ↑ | Discrete Gestures CLER ↓ | EPN-612 Acc | EPN-612 F1 | NinaPro DB5 Acc | NinaPro DB5 F1 | UCI-EMG Acc | UCI-EMG F1 |
|---|---|---|---|---|---|---|---|---|
| PhysioWave (4.8M) | 54.70±0.0 | 64.20±0.9 | 90.30±0.6 | 90.35±0.3 | 24.91±2.5 | 22.95±2.5 | 56.52±2.4 | 55.76±2.8 |
| TinyMyo (3.6M) | 39.70 | 64.20 | 84.68±1.9 | 84.68±1.9 | 25.26±4.5 | 23.29±4.9 | <u>85.99±0.4</u> | <u>85.66±0.4</u> |
| **NeuroRVQ-EMG** (5.9M) | **70.80±0.6** | **27.60±0.8** | **94.65±0.4** | **94.66±0.5** | **41.36±2.8** | **38.76±3.7** | **89.43±1.1** | **89.28±1.1** |

[Open-Source Benchmark Code Available](https://github.com/willraf/emg-fm-bench)

## Installation
```bash
conda create -n neurorvq python=3.10
conda activate neurorvq

# Install requirements
pip install -r requirements.txt
```

## Download Models
The models and the sample biosignal for reconstruction demos can be downloaded manually from [HuggingFace](https://huggingface.co/ntinosbarmpas/NeuroRVQ) or using python:
```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="ntinosbarmpas/NeuroRVQ", filename="pretrained_models/tokenizers/NeuroRVQ_EEG_tokenizer_v1.pt", local_dir="./")
hf_hub_download(repo_id="ntinosbarmpas/NeuroRVQ", filename="pretrained_models/foundation_models/NeuroRVQ_EEG_foundation_model_v1.pt", local_dir="./")
hf_hub_download(repo_id="ntinosbarmpas/NeuroRVQ", filename="pretrained_models/tokenizers/NeuroRVQ_EMG_tokenizer_v1.pt", local_dir="./")
hf_hub_download(repo_id="ntinosbarmpas/NeuroRVQ", filename="pretrained_models/foundation_models/NeuroRVQ_EMG_foundation_model_v1.pt", local_dir="./")
hf_hub_download(repo_id="ntinosbarmpas/NeuroRVQ", filename="example_files/eeg_sample/example_eeg_file.xdf", local_dir="./")
hf_hub_download(repo_id="ntinosbarmpas/NeuroRVQ", filename="pretrained_models/tokenizers/NeuroRVQ_ECG_tokenizer_v1.pt", local_dir="./")
hf_hub_download(repo_id="ntinosbarmpas/NeuroRVQ", filename="pretrained_models/foundation_models/NeuroRVQ_ECG_foundation_model_v1.pt", local_dir="./")
hf_hub_download(repo_id="ntinosbarmpas/NeuroRVQ", filename="pretrained_models/tokenizers/NeuroRVQ_PPG_tokenizer_v1.pt", local_dir="./")
```

## Model Loading / Usage

Load EEG tokenizer and see reconstruction results. Example for EEG tokenizer:
```python

from inference.run.NeuroRVQ_EEG_tokenizer_example import load_neurorqv_tokenizer

# Set run_example=True and plot_results=True to see reconstruction results
# Checkout the load_neurorqv_tokenizer() function to load and use tokenizer

load_neurorqv_tokenizer(run_example=True, plot_results=True, verbose=True, model_path='./pretrained_models/tokenizers/NeuroRVQ_EEG_tokenizer_v1.pt')
```

Load foundation model and see an example for fine-tuning. Example for EEG foundation model:
```python

from inference.run.NeuroRVQ_EEG_FM_example import load_neurorqv_fm

# Checkout the load_neurorqv_fm() function to see the correct model loading

load_neurorqv_fm(verbose=True, model_path = './pretrained_models/foundation_models/NeuroRVQ_EEG_foundation_model_v1.pt')
```

Load EMG tokenizer and see reconstruction results (downloads mini version of emg2pose). Example for EMG tokenizer:
```python

from inference.run.NeuroRVQ_EMG_tokenizer_example import load_neurorqv_tokenizer

# Set run_example=True and plot_results=True to see reconstruction results
# Checkout the load_neurorqv_tokenizer() function to load and use tokenizer

load_neurorqv_tokenizer(run_example=True, plot_results=True, verbose=True, model_path='./pretrained_models/tokenizers/NeuroRVQ_EMG_tokenizer_v1.pt')
```

Load foundation model and see an example for fine-tuning. Example for EMG foundation model:
```python

from inference.run.NeuroRVQ_EMG_FM_example import load_neurorqv_fm

# Checkout the load_neurorqv_fm() function to see the correct model loading

load_neurorqv_fm(verbose=True, model_path = './pretrained_models/foundation_models/NeuroRVQ_EMG_foundation_model_v1.pt')
```

Load ECG tokenizer and see reconstruction results (downloads and processes ptb-xl dataset). Example for ECG tokenizer:
```python

from inference.run.NeuroRVQ_ECG_tokenizer_example import load_neurorqv_tokenizer

# Set run_example=True and plot_results=True to see reconstruction results
# Checkout the load_neurorqv_tokenizer() function to load and use tokenizer

load_neurorqv_tokenizer(run_example=True, plot_results=True, verbose=True, model_path='./pretrained_models/tokenizers/NeuroRVQ_ECG_tokenizer_v1.pt')
```

Load foundation model and see an example for fine-tuning. Example for ECG foundation model:
```python

from inference.run.NeuroRVQ_ECG_FM_example import load_neurorqv_fm

# Checkout the load_neurorqv_fm() function to see the correct model loading

load_neurorqv_fm(verbose=True, model_path = './pretrained_models/foundation_models/NeuroRVQ_ECG_foundation_model_v1.pt')
```

Load PPG tokenizer and see reconstruction results (downloads and processes samples from the BIDMC PPG dataset). Example for PPG tokenizer:
```python

from inference.run.NeuroRVQ_PPG_tokenizer_example import load_neurorqv_tokenizer

# Set run_example=True and plot_results=True to see reconstruction results
# Checkout the load_neurorqv_tokenizer() function to load and use tokenizer

load_neurorqv_tokenizer(run_example=True, plot_results=True, verbose=True, model_path='./pretrained_models/tokenizers/NeuroRVQ_PPG_tokenizer_v1.pt')
```

## Citation
```
@misc{neurorvq,
      title={NeuroRVQ: Multi-Scale Biosignal Tokenization for Generative Foundation Models},
      author={Konstantinos Barmpas and Na Lee and Dimitrios Chalatsis and William Raftery and Yannis Panagakis and Dimitrios A. Adamos and Nikolaos Laskaris and Alexandros Koliousis and Dario Farina and Stefanos Zafeiriou},
      year={2026},
      eprint={2510.13068},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.13068},
}
```

