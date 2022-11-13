import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.losses import Loss, Reduction
from tensorflow.keras.activations import softplus
from typing import Tuple


class MaskedLoss(Loss):
    """
    自定义Loss的抽象类
    对于时间序列问题，我们要考虑输入的数据可能有目标值为Nan的情况，称为missing target
    面对这种missing value不应该计入loss的计算
    在之前的TimeSeriesTrain的预处理中，会对missing target进行填充，所以需要有一个mask_value（之前的填充值）
    """

    def __init__(self, mask_value, name):
        super(MaskedLoss, self).__init__(reduction=Reduction.AUTO, name=name)
        self._mask_value = mask_value

    def _mask_loss(self, loss_term, y_true):
        # missing target覆盖逻辑，如果真实值为0，那么mask的对应位置为False，就不计入loss计算
        mask = tf.not_equal(y_true, tf.constant(self._mask_value, dtype=tf.float32))
        mask = tf.dtypes.cast(mask, tf.float32)
        return tf.multiply(loss_term, mask)

    def call(self, y_true, y_pred_bundle):
        raise NotImplementedError


class GaussianLogLikelihood(MaskedLoss):
    """
    自定义高斯分布的loss
    """

    def __init__(self, mask_value=0, name="gaussian_log_likelihood"):
        super(GaussianLogLikelihood, self).__init__(mask_value, name)

    def call(self, y_true, y_pred_bundle):
        """
        真实标签服从高斯分布的负对数似然损失

        Parameters
        ----------
        y_true
        y_pred_bundle: Tuple[tf.Tensor, tf.Tensor] -- (mu, sigma)

        Returns
        -------

        """
        # mu = y_pred[:, 0, 0]
        # sigma = y_pred[:, 0, 1] + 1e-4
        # y_true = tf.cast(y_true, tf.float32)
        # square = tf.square(mu - y_true)  ## preserve the same shape as y_pred.shape
        # ms = tf.divide(square, sigma) + K.log(sigma)
        # ms = tf.reduce_mean(ms)
        # return ms
        mu, sigma = y_pred_bundle
        batch_size = mu.shape[0]

        # loss
        sigma += 1e-4  # 防止sigma为0 影响loss计算
        loss_term = 0.5 * tf.math.log(sigma) + 0.5 * tf.divide(tf.square(y_true - mu), sigma)

        # mask
        masked_loss_term = self._mask_loss(loss_term, y_true)
        # divide by batch size bc auto reduction will sum over batch size
        return masked_loss_term / batch_size


class NegativeBinomialLogLikelihood(MaskedLoss):
    def __init__(self, mask_value=0, name="negative_binomial_log_likelihood"):
        super(NegativeBinomialLogLikelihood, self).__init__(mask_value, name)

    def call(self, y_true, y_pred_bundle):
        mu, alpha = y_pred_bundle
        batch_size = mu.shape[0]
        # loss 计算，按照deepar 原始论文第四页进行计算
        alpha_y_pred = tf.multiply(alpha, mu)
        alpha_div = tf.divide(1.0, alpha)
        denom = tf.math.log(1 + alpha_y_pred)
        log_loss = (
                tf.math.lgamma(y_true + alpha_div)
                - tf.math.lgamma(y_true + 1.0)
                - tf.math.lgamma(alpha_div)
        )
        loss_term = (
                log_loss
                - tf.divide(denom, alpha)
                + tf.multiply(y_true, tf.math.log(alpha_y_pred) - denom)
        )
        loss_term = -loss_term
        # mask
        masked_loss_term = self._mask_loss(loss_term, y_true)
        # divide by batch size bc auto reduction will sum over batch size
        return masked_loss_term / batch_size


def negative_binomial_sampling(mu, sigma, samples=1):
    """
    https://ts.gluon.ai/stable/_modules/gluonts/mx/distribution/neg_binomial.html#NegativeBinomial

    """
    tol = 1e-5
    r = 1.0 / sigma
    theta = sigma * mu
    r = tf.math.minimum(tf.math.maximum(tol, r), 1e10)
    theta = tf.math.minimum(tf.math.maximum(tol, theta), 1e10)
    p = 1 / (theta + 1)
    return np.random.negative_binomial(r, p, samples)
