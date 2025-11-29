'''
Wrapper classes of foundation model modules for use in main benchmarking script
'''
import torch
from abc import ABC, abstractmethod
from fine_tuning.NeuroRVQ_EEG_FM_FineTuning import NeuroRVQModule

class FinetuningWrapper(ABC):
    """
    Wrapper class for initializing model, fitting and evaluating on benchmark data, and storing results
    """
    def __init__(self):
        self.model = None
        self.results = {}

    @abstractmethod
    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        print("fit function not implemented")

    def size(self):
        """ Returns number of trainable parameters in model """
        if self.model is None:
            print("model not initialised")
        else:
            return self.model.size()

class NeuroRVQWrapper(FinetuningWrapper):
    def __init__(self, n_time, ch_names, n_outputs, train_head_only, args, foundation_model):
        super().__init__()
        self.model = NeuroRVQModule(
            sample_length=n_time, 
            chnames=ch_names,
            n_out=n_outputs,
            train_head_only=train_head_only,
            args = args,
            foundation_model = foundation_model
            )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        self.model.fit(train_dataset, validation_dataset, batch_size, epochs)
        self.results = self.model.results
        
    def save_model(self, path):
        print(f'Saving checkpoint to {path}...')
        torch.save(self.model.model.state_dict(), path)
