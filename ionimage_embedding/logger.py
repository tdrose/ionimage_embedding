from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
import collections

class DictLogger(Logger):
    """
    Minimal Logger to save all recorded metrics during training into a dictionary
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