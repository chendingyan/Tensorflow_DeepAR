import logging
import tensorflow as tf
logger = logging.getLogger(__name__)

class EarlyStopping(object):
    """ class that monitors a metric and stops training when the metric has stopped improving

        Keyword Arguments:
            monitor_increase {bool} -- if True, stops training when metric stops increasing. If False,
                when metric stops decreasing (default: {False})
            patience {int} -- after how many epochs of degrading performance should training be stopped (default: {0})
            delta {int} -- within what range should a change in performance be evaluated, the comparison to
                determine stopping will be previous metric vs. new metric +- stopping_delta(default: {0})
            active {bool} -- whether early stopping callback is active or not (default: {True})
    """

    def __init__(
        self,
        monitor_increase: bool=False,
        patience: int=0,
        delta: int=0,
        active: bool=True
    ):

        self._monitor_increase = monitor_increase
        self._patience = patience
        self._best_metric = None
        self._degrade_count = 0
        self._active = active
        self._delta = delta

    def __call__(
        self,
        cur_metric: tf.Tensor
    ) -> bool:

        # check for base cases
        if not self._active:
            return False
        elif self._best_metric is None:
            self._best_metric = cur_metric
            return False

        # update degrade_count according to parameters
        else:
            if self._monitor_increase:
                if cur_metric < self._best_metric + self._delta:
                    self._degrade_count += 1
                else:
                    self._best_metric = cur_metric
                    self._degrade_count = 0
            else:
                if cur_metric > self._best_metric - self._delta:
                    self._degrade_count += 1
                else:
                    self._best_metric = cur_metric
                    self._degrade_count = 0

        # check for early stopping criterion
        if self._degrade_count >= self._patience:
            logger.info(f'Metric has degraded for {self._degrade_count} epochs, exiting training')
            return True
        else:
            return False

    # def _callbacks(self, filepath, early_stopping = True, val_set = True, stopping_patience = 0, scheduler_factor = 0.2,
    #         scheduler_patience = 5, min_cosine_lr = 0):
    #     """ defines callbacks that we want to use during fitting
    #             :param filepath: filepath to save checkpoint and tensorboard files
    #             :param early_stopping: whether to include early stopping callback
    #             :param val_set: boolean, whether validation set exists
    #             :param stopping_patience: early stopping callback patience, default 0
    #             :param scheduler_factor: plateau lr scheduler decrease factor, default 0.2
    #             :param scheduler_patience: plateau lr scheduler patience, default 5
    #             :param min_cosine_lr: cosine annealing lr scheduler minimum lr, default 0
    #             :return: list of callbacks for fitting
    #     """

    #     if not val_set:
    #         monitor = 'loss'
    #     else:
    #         monitor = 'val_loss'

    #     callbacks = []

    #     # Model Checkpoint
    #     checkpoint = ModelCheckpoint(filepath,
    #         monitor=monitor,
    #         verbose=self.verbose,
    #         save_best_only = True)

    #     # Tensorboard
    #     tb = TensorBoard(log_dir=filepath)

    #     callbacks = [checkpoint, tb]

    #     # Early Stopping
    #     if early_stopping:
    #         es = EarlyStopping(monitor=monitor,
    #             patience = stopping_patience,
    #             verbose = self.verbose)
    #         callbacks.append(es)

    #     # Learning Rate Scheduler (by default simply use Adam)
    #     if self.scheduler == 'plateau':
    #         scheduler = ReduceLROnPlateau(monitor=monitor,
    #             factor = scheduler_factor,
    #             patience=scheduler_patience,
    #             verbose=self.verbose)
    #         callbacks.append(scheduler)

    #     elif self.scheduler == 'cosine_annealing':

    #         # custom cosine annealing schedule
    #         def cosine_annealing(epoch, cur_lr):
    #             return min_cosine_lr + (self.lr - min_cosine_lr) * (1 + math.cos(math.pi * epoch / self.epochs)) / 2

    #         scheduler = LearningRateScheduler(cosine_annealing,
    #             verbose=self.verbose)
    #         callbacks.append(scheduler)
    #     return callbacks