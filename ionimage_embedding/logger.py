import pandas as pd
import collections

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only


class DictLogger(Logger):
    """
    Minimal Logger for logging recorded metrics of a lightning model 
    during training into a dictionary.
    """
    
    def __init__(self):
        super(DictLogger).__init__()

        self.logged_metrics = collections.defaultdict(list)
    
    @property
    def name(self):
        return "DictLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here

        for key, val in metrics.items():
            self.logged_metrics[key].append(val)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Access metrics at the end of each training epoch
        metrics = trainer.callback_metrics
        step = trainer.global_step
        self.log_metrics(metrics, step)

    def on_train_epoch_end(self, trainer, pl_module):
        # Access metrics at the end of each training epoch
        metrics = trainer.callback_metrics
        step = trainer.global_step
        self.log_metrics(metrics, step)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass


class PerformanceLogger:
    """
    Custom logger for storing performance metrics of different models.
    Is used to log model performance on the test dataset across different
    models and bootstraps.
    """
    
    def __init__(self, *args, verbose: bool=False):

        self.keys = args
        self.verbose = verbose
        
        self.data = {k: [] for k in self.keys}

    def add_result(self, *args):
        if len(args) != len(self.keys):
            raise ValueError('Number of arguments does not match number of keys')
        for i, k in enumerate(self.keys):
            self.data[k].append(args[i])
        
        if self.verbose:
            print('Added result:')
            self.print_last()
            print('')

    def get_keys(self):
        return self.keys

    def get_df(self):
        return pd.DataFrame(self.data)
    
    def print_last(self):
        for k in self.data:
            print(f'{k}: {self.data[k][-1]}')
