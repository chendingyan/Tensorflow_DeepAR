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
            monitor_increase: bool = False,
            patience: int = 0,
            delta: int = 0,
            active: bool = True
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
