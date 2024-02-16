import pandas as pd
import collections

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only


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

class PerformanceLogger:
    
    def __init__(self, scenario: str='Scenario',metric: str='Accuracy', 
                 evaluation: str='Evaluation', fraction: str='Fraction'):
        self.scenario_l = []
        self.metric_l = []
        self.eval_l = []
        self.frac_l = []

        self.scenario = scenario
        self.metric = metric
        self.evaluation = evaluation
        self.fraction = fraction

    def add_result(self, scenario: str, metric: float, evaluation: str, fraction: float):
        self.scenario_l.append(scenario)
        self.metric_l.append(metric)
        self.eval_l.append(evaluation)
        self.frac_l.append(fraction)

    def get_df(self):
        return pd.DataFrame({self.scenario: self.scenario_l, 
                             self.metric: self.metric_l, 
                             self.evaluation: self.eval_l, 
                             self.fraction: self.frac_l})
