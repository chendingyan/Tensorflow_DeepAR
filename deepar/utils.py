import numpy as np
import tensorflow as tf


def detect_date_consecutive(data, date_col, freq):
    """
    日期不连续，导致diff的结果unique不同
    """
    temp = (round(data[date_col].diff() / np.timedelta64(1, freq))).reset_index()
    if temp[1:][date_col].nunique() > 1:
        return False
    return True


def detect_date_freq(data, date_col, freq):
    """
    日期freq和真实的日期对不上
    """
    temp = (round(data[date_col].diff() / np.timedelta64(1, freq))).reset_index()
    if temp[1:][date_col].unique()[0] != 1:
        return False
    return True


def build_tf_lookup_hashtable(scale_data):
    """
    创建一个tf的hash table，保存预测值的放缩变换
    key 为 scale data的index
    value 为 scale data的value
    """
    return tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(scale_data.index.values, dtype=tf.int32),
            values=tf.constant(scale_data.values, dtype=tf.float32),
        ),
        default_value=tf.constant(-1, dtype=tf.float32),
    )


def unscale(mu, sigma, scale_keys, lookup_hashtable):
    """
    unscale prediction
    """
    scale_values = tf.expand_dims(lookup_hashtable.lookup(scale_keys), 1)
    unscaled_mu = tf.multiply(mu, scale_values)
    unscaled_sigma = tf.divide(sigma, tf.sqrt(scale_values))
    return unscaled_mu, unscaled_sigma
