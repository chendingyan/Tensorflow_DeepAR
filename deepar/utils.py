import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import timedelta
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


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


WEEKDAY_MAP = {
    0: "W-MON", 1: "W-TUE", 2: "W-WED", 3: "W-THU", 4: "W-FRI", 5: "W-SAT", 6: "W-SUN"
}


def date_filler(base_df, date_col, groupby, freq='D', how='local_min_max'):
    """
    日期连续性填充，会根据`freq`的要求填充DataFrame日期

    Parameters
    ----------
    base_df: pd.DataFrame 数据集
    date_col: str 需要填充的日期列
    groupby: Optional[list[str]] 填充日期时候的组别 可以为None，比如只有一条时间序列的时候
    freq: str, optional (default='D')
        填充颗粒度，'D', 'W-MON', 'W-TUE', ...参考pandas的DateOffset定义:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    how: str, optional(default='local_min_max')
        填充方式：
        cartesian_product 按照数据集日期最大值和最小值及笛卡尔积`groupby`对所有pair进行填充

        global_min_max 按照数据集日期最大最小值对`groupby`后每一个group的日期进行填充

        local_min_global_max 按照数据集日期最大值及`groupby`后各组日期最小值分别对各组进行填充

        local_min_max 按照`groupby`后各组日期最大最小值对各组分别进行填充
    """

    def _set_freq(base_df, freq, date_col):
        if freq == 'W':
            if base_df[date_col].dt.weekday.nunique() != 1:
                raise ValueError(f"Weekday are not unique for {date_col}!")
            freq = WEEKDAY_MAP[base_df[date_col].min().weekday()]
        return freq

    base_df[date_col] = pd.to_datetime(base_df[date_col])
    freq = _set_freq(base_df, freq, date_col)

    if groupby:
        if how == 'cartesian_product':
            dim_index = [date_col] + groupby
            return base_df.set_index(dim_index).unstack(groupby).asfreq(freq).stack(
                groupby, dropna=False).sort_index(level=groupby).reset_index()
        else:
            valid_pairs = _get_valid_pairs(base_df, date_col, groupby, freq)
            if how == 'local_min_max':
                valid_pairs = valid_pairs[(valid_pairs[f'min_{date_col}'] <= valid_pairs[date_col]) &
                                          (valid_pairs[f'max_{date_col}'] >= valid_pairs[date_col])]
            elif how == 'local_min_global_max':
                valid_pairs = valid_pairs[(valid_pairs[f'min_{date_col}'] <= valid_pairs[date_col])]
            elif how == 'global_min_max':
                pass
            else:
                raise ValueError(f'Currently not supporting for {how} as how key!')

            valid_pairs = valid_pairs.drop(columns=[f'min_{date_col}', f'max_{date_col}'])
            return_df = valid_pairs.merge(base_df, on=groupby + [date_col], how='left').reset_index(
                drop=True)
            return return_df
    else:
        return base_df.set_index(date_col).asfreq(freq).reset_index()


def _get_valid_pairs(base_df, date_col, groupby, freq):
    merge_col = 'key'
    while merge_col in base_df.columns:
        merge_col += "_1"
    date_index = pd.date_range(start=base_df[date_col].min(), end=base_df[date_col].max(),
                               freq=freq).to_frame().reset_index(drop=True).rename(columns={0: date_col})
    date_index[merge_col] = 1
    valid_pairs = base_df.groupby(groupby).agg({date_col: ['min', 'max']}).reset_index()
    valid_pairs.columns = groupby + [f'min_{date_col}', f'max_{date_col}']
    valid_pairs[merge_col] = 1
    valid_pairs = valid_pairs.merge(date_index, on=[merge_col])
    return valid_pairs.drop(columns=[merge_col])


def date_aggregator(base_df, date_col, groupby, freq, agg, label):
    """
    将DataFrame根据`freq`进行聚合

    Parameters
    ----------
    base_df: pd.Dataframe 数据集
    date_col: str 日期列
    groupby: list[str] 聚合日期时候的组别
    agg: dict
        对除`date_col`和`groupby`外的特定列的聚合操作，例如聚合销量{'sales' : 'sum'}
    freq: str, optional (default='D')
        填充颗粒度，'D', 'W-MON', 'W-TUE', ...参考pandas的DateOffset定义:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    label: str, optional (default='left')
        区间边界的标记, left or right
        参考pandas.Grouper的label定义: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Grouper.html

    Examples
    ----------
        freq = 'W', label = 'right' 会以一周的结束的那一天作为聚合的日期，比如聚合后结果为2021-03-07(周日)代表的是
        2021-03-01(周一)到2021-03-07(周日)的聚合结果。
        freq = 'W', label = 'left' 会以一周的开始的那一天作为聚合的日期，比如聚合后结果为2021-03-07(周日)代表的是
        2021-03-07(周日)到2021-03-13(周六)的聚合结果

    """

    def _set_freq(base_df, freq, label):
        if freq == 'W' and label == 'right':
            freq = WEEKDAY_MAP[base_df[date_col].max().weekday()]
        elif freq == 'W' and label == 'left':
            freq = WEEKDAY_MAP[(base_df[date_col].max() + timedelta(days=1)).weekday()]
        elif freq == 'M' and label == 'left':
            freq = 'MS'
        elif freq == 'M' and label == 'right':
            freq = 'M'
        return freq

    base_df[date_col] = pd.to_datetime(base_df[date_col])
    freq = _set_freq(base_df, freq, label)

    return base_df.groupby(
        groupby + [pd.Grouper(key=date_col, freq=freq, label=label, closed=label)]).agg(agg).reset_index()


def second_of_minute(index):
    return index.second.values


def second_of_minute_index(index):
    return index.second.astype(float).values


def minute_of_hour(index: pd.PeriodIndex) -> np.ndarray:
    return index.minute.values


def minute_of_hour_index(index: pd.PeriodIndex) -> np.ndarray:
    return index.minute.astype(float).values


def hour_of_day(index: pd.PeriodIndex) -> np.ndarray:
    return index.hour.values


def hour_of_day_index(index: pd.PeriodIndex) -> np.ndarray:
    return index.hour.astype(float).values


def day_of_week(index: pd.PeriodIndex) -> np.ndarray:
    return index.dayofweek.values


def day_of_week_index(index: pd.PeriodIndex) -> np.ndarray:
    return index.dayofweek.astype(float).values


def day_of_month(index: pd.PeriodIndex) -> np.ndarray:
    return (index.day.values - 1)


def day_of_month_index(index: pd.PeriodIndex) -> np.ndarray:
    return index.day.astype(float).values


def day_of_year(index: pd.PeriodIndex) -> np.ndarray:
    return (index.dayofyear.values - 1)


def day_of_year_index(index: pd.PeriodIndex) -> np.ndarray:
    return index.dayofyear.astype(float).values


def month_of_year(index: pd.PeriodIndex) -> np.ndarray:
    return index.month.values


def month_of_year_index(index: pd.PeriodIndex) -> np.ndarray:
    return index.month.astype(float).values


def week_of_year(index: pd.PeriodIndex) -> np.ndarray:
    # * pandas >= 1.1 does not support `.week`
    # * pandas == 1.0 does not support `.isocalendar()`
    # as soon as we drop support for `pandas == 1.0`, we should remove this
    try:
        week = index.isocalendar().week
    except AttributeError:
        week = index.week
    return week.astype(float).values


def week_of_year_index(index: pd.PeriodIndex) -> np.ndarray:
    try:
        week = index.isocalendar().week
    except AttributeError:
        week = index.week
    return week.astype(float).values


def time_features_from_frequency_str(freq_str: str):
    features_by_offsets = {
        offsets.YearBegin: [],
        offsets.YearEnd: [],
        offsets.QuarterBegin: [month_of_year],
        offsets.QuarterEnd: [month_of_year],
        offsets.MonthBegin: [month_of_year],
        offsets.MonthEnd: [month_of_year],
        offsets.Week: [day_of_month, week_of_year],
        offsets.Day: [day_of_week, day_of_month, day_of_year],
        offsets.BusinessDay: [day_of_week, day_of_month, day_of_year],
        offsets.Hour: [hour_of_day, day_of_week, day_of_month, day_of_year],
        offsets.Minute: [
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
        ],
        offsets.Second: [
            second_of_minute,
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, features in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return features

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}

    The following frequencies are supported:

        Y   - yearly
            alias: A
        Q   - quarterly
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)
