from abc import ABC

from pandas.core.dtypes.common import is_datetime64_any_dtype, is_numeric_dtype

from deepar.exceptions import DataSchemaCheckException, ParameterException
from deepar.utils import detect_date_consecutive, detect_date_freq, time_features_from_frequency_str

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dropout, Bidirectional, LSTM, Dense, TimeDistributed, Softmax, Multiply, \
    Lambda, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dataset(ABC):
    def __init__(self):
        super().__init__()

    def next_batch(self, **kwargs):
        pass


class TSTrainDataset(Dataset):
    def __init__(self, df, date_col, target_col, groupby_col, freq, feat_dynamic_reals=None, feat_static_cats=None,
                 feat_static_reals=None, lags=None, valid_split=0,
                 count_data=True):
        """

        Parameters
        ----------
        df: 训练集
        date_col: 日期列，不能是date_index（pytorch-forecasting里的是需要index），需要是datetime格式的
        target_col：目标列，需要是数值类型
        groupby_col: 唯一的groupby,类比与gluonts里的ITEM_ID
        freq：pandas freq标准，可参考https://pandas.pydata.org/docs/user_guide/timeseries.html
        feat_dynamic_reals：随着时间变化会变化的数值特征，比如价格
        feat_static_cats：随着时间变化不变的类别特征，比如城市(需要包括groupby_col)
        feat_static_reals：随着时间变化不变的数值特征
        lags：是否做lag，lag的话做多少
        valid_split: 最后多少的数据集会作为Validation
        count_data: 是否为实数预测，如果是True代表预测结果会是自然数，并且用负二项分布
        """
        self.df = df
        self.freq = freq
        self.date_col = date_col
        self.target_col = target_col
        self.groupby_col = groupby_col
        self.cont_feats = feat_dynamic_reals if feat_dynamic_reals else []
        self.cat_feats = feat_static_cats if feat_static_cats else []
        self.static_real_feats = feat_static_reals if feat_static_reals else []
        self.valid_split = valid_split
        self.lags = lags
        # 如果是整数预测，用负二项分布，则是count_data
        self._count_data = count_data
        self._label_encoder = None
        self._encoder_dict = dict()
        self._standard_scaler = None
        self._target_means = None
        self.df = self._data_check(self.df)
        self._clean_features()
        self._set_mask_value()
        # self.df = self._add_lag_features(self.df)
        self._add_date_features()
        self._add_age_features()
        # 增加lags，更新cont_feats
        self._make_cat_encoder()
        self._train_val_split()
        self._store_target_means()
        self._standardize()
        self._record_missing_target_values()

    def _data_check(self, data):
        """
        进行数据检查，内容包括
        1. 主键是否有空值 + 主键是否唯一 (主键应该包括日期列)
        2. 日期列是否为日期类型
        3. 目标列是否为数值类型
        4. 时间颗粒度Freq和数据集的颗粒度是否一致
        5. 每个groupby pair下预测开始时间之前的历史数据是否存在
        6. 每个groupby pair下test set是否提前建立

        Parameters
        ----------
        data: pd.DataFrame

        Raises
        -------
        DataSchemaCheckException if data check fails
        """
        if self.groupby_col is None:
            data['category'] = 'dummy_cat'
            self.groupby_col = 'category'
        data[self.groupby_col] = data[self.groupby_col].astype(str)
        primary_key = [self.date_col, self.groupby_col]
        data = data.sort_values(by=primary_key)
        if data[primary_key].isnull().values.any():
            raise DataSchemaCheckException(f'输入主键 {primary_key} 存在空值（Nan），请提前检查主键中是否有空值')
        if len(data.drop_duplicates(subset=primary_key)) != len(data):
            raise DataSchemaCheckException(f'输入主键 {primary_key} 并不唯一，请重新检查输入主键的正确性')
        if not is_datetime64_any_dtype(data[self.date_col]):
            data[self.date_col] = pd.to_datetime(data[self.date_col])
        if self.target_col and not is_numeric_dtype(data[self.target_col]):
            raise DataSchemaCheckException(f'目标列 {self.target_col} 并不是一个数值列，预测仅支持数值类预测，请检查目标列是否设置有误')
        date_consecutive_check = data.groupby(self.groupby_col).apply(
            lambda x: detect_date_consecutive(x, self.date_col, self.freq)).all()
        if not date_consecutive_check:
            raise DataSchemaCheckException(
                f'日期列 {self.date_col} 不连续，应该存在缺失时期，需要提前对所有的日期进行填充，保证在既定颗粒度{self.freq}下没有缺失时间点')
        date_freq_check = data.groupby(self.groupby_col).apply(
            lambda x: detect_date_freq(x, self.date_col, self.freq)).all()
        if not date_freq_check:
            raise DataSchemaCheckException(f'日期列时间颗粒度 {self.date_col} 和既定颗粒度{self.freq} 不一致')
        return data

    def _clean_features(self):
        self.df = self.df[
            self.cont_feats + self.cat_feats + self.static_real_feats + [self.date_col] + [self.target_col]]
        # 把目标列定死为target，其实没必要
        self.df = self.df.rename(columns={self.target_col: 'target'})
        self._raw_df = self.df.copy()
        self.target_col = 'target'

    def _add_lag_features(self, df):
        """

        Returns
        -------

        """
        if self.lags is not None:
            self.lag_dict = {i: self.target_col for i in self.lags}
            groupby = df.groupby(self.groupby_col)
            for idx in self.lag_dict:
                df_add = groupby[self.lag_dict[idx]].shift(idx)
                new_col_name = f"{self.target_col}_lag_{idx}"
                df[new_col_name] = df_add
                # TODO: fillna的方式 0/target means等未来都可以尝试拓展
                df[new_col_name] = df[new_col_name].interpolate(limit_direction='both')
                self.cont_feats.append(new_col_name)
        return df

    def _add_date_features(self):
        """
        增加日期相关的特征，这里增加的特征和Gluonts的方法保持一致
        """
        new_feats = []
        func_list = time_features_from_frequency_str(self.freq)
        for func in func_list:
            new_feats.append(func.__name__)
            self.df[func.__name__] = func(self.df[self.date_col].dt)
        # self.df["year"] = self.df[self.date_col].dt.year
        # self.df["month"] = self.df[self.date_col].dt.month
        # self.df["day"] = self.df[self.date_col].dt.day
        # self.df["hour"] = self.df[self.date_col].dt.hour
        self.df.drop(columns=[self.date_col], inplace=True)
        # new_feats = ['year', 'month', 'day', 'hour']
        for feat in new_feats:
            if feat not in self.cont_feats:
                self.cont_feats.append(feat)

    def _add_age_features(self):
        # 增加age，论文中说这是一个重要的特征，代表从时间序列第一个时间点到现在经历了多久
        self.df['_age'] = self.df.groupby(self.groupby_col).cumcount()
        # 记录一下训练集里每条时间序列的age的最大值有多大
        self._train_set_ages = self.df.groupby(self.groupby_col)['_age'].agg('max')
        # 由于对训练集的groupby col进行encoder的时候 增加了一个dummy_test_category的值，所以需要增加一下对应的age
        self._train_set_ages['dummy_test_category'] = 0
        self.cont_feats.append('_age')

    def _make_cat_encoder(self):
        """
        对category的groupby进行编码，这里目前是写死了LabelEncoder
        尝试过在初始化的时候传入category_encoders库中的任意编码器,但是由于API返回的结果略有不同，代码通用性和可读性会下降，舍弃了这种写法
        """
        # need embeddings for cats in val, even if not in train
        # 1 extra for test cats not included in train or val
        self.ts_unique_values = self.df[self.groupby_col].unique()
        self._num_cats = self.df[self.groupby_col].nunique() + 1
        self.df[self.cat_feats] = self.df[self.cat_feats].astype(str)
        for cat in self.cat_feats:
            label_encoder = LabelEncoder()
            cat_names = self.df[cat].append(pd.Series(['dummy_test_category']))
            label_encoder.fit(cat_names)
            self.df[cat] = label_encoder.transform(self.df[cat])
            self._encoder_dict[cat] = label_encoder
        self._label_encoder = self._encoder_dict[self.groupby_col]

    def _train_val_split(self):
        if self.valid_split == 0:
            self._train_data = self.df.copy()
        else:
            whole_data_len = len(self.df)
            n_rows = int(whole_data_len * self.valid_split)
            self._train_data = self.df.head(whole_data_len - n_rows)
            self._val_data = self.df.tail(n_rows)

    def _store_target_means(self):
        """
        计算加权采样的V
        储存下目标值的平均值，为了后面的一些填充的时候用
        """
        # store target means over training set
        # _target_means代表加权采样的放缩系数v
        # self._target_means代表每个group下的目标值的平均值+1
        self._target_means = 1 + self._train_data.groupby(self.groupby_col)['target'].agg('mean')

        target_mean = 1 + self._train_data['target'].dropna().mean()
        self._create_sampling_distribution()
        # add 'dummy_test_category' as key to target means
        self._target_means.loc[self._label_encoder.transform(['dummy_test_category'])[0]] = target_mean
        # 如果在Validation中的groupby value 在训练集中不存在，使用训练集drop na后的mean填充
        if self.valid_split != 0:
            # if group in val doesn't exist in train, standardize by overall mean
            for group in self._val_data[self.groupby_col].unique():
                if group not in self._train_data[self.groupby_col].unique():
                    self._target_means.loc[group] = target_mean

    def _create_sampling_distribution(self):
        """
        创建辅助np.random.choice中选择的概率分布p
        方式是根据softmax函数normalize 放缩系数v，然后根据v来采样
        按照序列绝对值的占比，如果序列越大 采样到的概率越大
        """
        scale_factor_v = self._target_means
        # softmax
        e_x = np.exp(scale_factor_v - np.max(scale_factor_v))
        self._scale_factors_softmax = e_x / e_x.sum(axis=0)

    def _standardize(self):
        """
        论文里说需要标准化到mean = 0， std = 1
        除了datetime字段，groupby字段和目标字段都要被标准化
        """
        covariate_names = ['target', self.groupby_col] + list(self._encoder_dict.keys())
        covariate_mask = [False if col_name in covariate_names else True for col_name in self.df.columns]
        self._standard_scaler = StandardScaler()
        self._train_data.loc[:, covariate_mask] = self._standard_scaler.fit_transform(
            self._train_data.loc[:, covariate_mask].astype('float'))
        if self.valid_split != 0:
            # self._val_data = self._val_data.reset_index(drop=True)
            self._val_data.loc[:, covariate_mask] = self._standard_scaler.transform(
                self._val_data.loc[:, covariate_mask].astype('float'))
        self.df.loc[:, covariate_mask] = self._standard_scaler.transform(self.df.loc[:, covariate_mask].astype('float'))

    def _record_missing_target_values(self):
        """
        记录数据中目标值为空的地方
        """
        # missing_tgt_vals是一个字典，key为空target值所对应的groupby col，value为这个groupby col 空值对应的_age
        self._missing_tgt_vals = {}
        self._mask_missing_targets(self._train_data)
        if self.valid_split != 0:
            self._val_data = self._val_data.reset_index(drop=True)
            self._mask_missing_targets(self._val_data)

    def _mask_missing_targets(self, df):
        for idx in pd.isnull(df)['target'].to_numpy().nonzero()[0]:
            # 找到所有target为空的行idx数
            groupby_key = df[self.groupby_col].iloc[idx]
            if groupby_key in self._missing_tgt_vals.keys():
                self._missing_tgt_vals[groupby_key].append(df['_age'].iloc[idx])
            else:
                self._missing_tgt_vals[groupby_key] = [df['_age'].iloc[idx]]

    def _set_mask_value(self):
        """
        为后面训练队时候对missing target填入mask value做准备
        """
        # set mask value
        if self._count_data:
            self._mask_value = 0
        else:
            self._mask_value = self.df['target'].min() - 1

    def next_batch(self, model, batch_size, window_size, valid_set=False):
        """
        每次拿到下一个batch的训练集数据，为generator的辅助函数

        构建batch的过程是
        1. 根据groupby col，随机抽取batch size条时间序列
        2. 遍历这些时间序列 构建
        Parameters
        ----------
        model
        batch_size
        window_size
        valid_set: 如果True的话会在valid data里面去采样出batch
        kwargs

        Returns
        -------

        """
        if valid_set:
            assert self._val_data is not None, "Asking for validation batch, but validation split was 0 in object construction"
            cat_samples = np.random.choice(self._val_data[self.groupby_col].unique(), batch_size)
            data = self._val_data
        else:
            cat_samples = np.random.choice(self._train_data[self.groupby_col].unique(), batch_size,
                                           p=self._scale_factors_softmax)
            data = self._train_data
        sampled = []
        # cat_samples实际上就是被选中的时间序列
        # 对采样到的时间序列增加prev_target
        for cat in cat_samples:
            cat_data = data[data[self.groupby_col] == cat]
            if valid_set:
                cat_data = self._add_prev_target_col(cat_data,
                                                     self._train_data[self._train_data[self.groupby_col]] == cat)
            else:
                cat_data = self._add_prev_target_col(cat_data)

            cat_data['target'] = cat_data['target'].fillna(
                self.mask_value)  # 如果是fillna进去的target值，在后面计算loss的时候都会被标记并且不进行计算
            # 如果该时间序列的长度大于window size，需要采样到window_size长度
            if len(cat_data) > window_size:
                sampled_cat_data = self._sample_ts(cat_data, window_size)
                # print(f"window size < len(time series):{sampled_cat_data.shape}")
            else:
                sampled_cat_data = cat_data
                print(f"window size > len(time series):{sampled_cat_data.shape}")

            sampled.append(sampled_cat_data)
        # 窗口采样完的数据拼接起来，然后产生batch输出
        data = pd.concat(sampled)
        if 'prev_target' not in self.cont_feats:
            self.cont_feats.append('prev_target')
        cont_inputs = tf.constant(
            data[self.cont_feats].values.reshape(batch_size, window_size, -1),
            dtype=tf.float32)

        # print(f"cont input: {cont_inputs.shape}")
        cat_inputs = []
        for cat in self.cat_feats + self.static_real_feats:
            cat_input = tf.constant(data[cat].values.reshape(batch_size, window_size, -1), dtype=tf.float32)
            cat_inputs.append(cat_input)

        # cat_labels = tf.constant(cat_samples.reshape(batch_size, 1), dtype=tf.int32)
        scale_values = tf.constant(self._target_means[cat_samples].values, dtype=tf.float32)
        targets = tf.constant(data['target'].values.reshape(batch_size, window_size, 1), dtype=tf.float32)
        return [cont_inputs] + cat_inputs, scale_values, targets

    def _add_prev_target_col(self, df, train_df=None):
        """
        根据论文，需要增加prev target(lag1)进行自回归
        df 就是某一条时间序列
        train_df 是该时间序列在训练集的部分
        target_means 放缩系数v
        """
        target_means = self._target_means
        df = df.reset_index(drop=True)
        # means 代表把整个df的对应时间序列的target means都填上， target_means是一个series，包括每条时间序列（groupby）的放缩系数v，假设一个dummy category
        # 然后df是整个数据集，就会把对应位置的mean都取到
        means = target_means[df[self.groupby_col]].reset_index(drop=True)
        if train_df is None:
            # 这类情况是在采样训练集的batch，那么不会传入train_df，直接采样自己
            df['prev_target'] = df['target'].shift(1).fillna(0)
        else:
            # val or test data
            if 'target' in df.columns:
                # 这类情况是在采样验证集的batch，需要传入验证集对应的训练集的该条时间序列
                # 然后prev target要lag 1，第一个位置能取到训练集最后一个target
                if train_df.shape[0] > 0:
                    df['prev_target'] = df['target'].shift(1).fillna(train_df['target'].dropna().tail(1))
                else:
                    df['prev_target'] = df['target'].shift(1).fillna(0)
            else:
                # 测试集 直接复制一波训练集最后一个target作为prev target
                df['prev_target'] = train_df['target'].dropna().tail(1).repeat(repeats=df.shape[0]).reset_index(
                    drop=True)
        # 放缩prev_target
        # self._scale_prev_target_col(df, means)
        # 其实应该不太需要插值
        df['prev_target'] = df['prev_target'].interpolate(limit_direction='both')
        return df

    def _scale_prev_target_col(self, df, means):
        """
        根据放缩系数v，把prev_target列也放缩一下
        """
        if (means == 0).all():
            df['prev_target'] = means
        else:
            df['prev_target'] = df['prev_target'] / means
        return df

    def _sample_ts(self, df, sample_length):
        """
        采样时间序列
        随机选取开始的时间点
        """
        start_index = np.random.choice([i for i in range(0, len(df) - sample_length)])
        return df.iloc[start_index: start_index + sample_length, ]

    def _sample_missing_prev_targets(self, df, full_df_ages, model, window_size, batch_size, training=True):
        """
        如果输出的目标值中不含有空值 NAN，那么应该一辈子都不会调用这个函数
        Parameters
        ----------
        df
        full_df_ages
        model
        window_size
        batch_size
        training

        Returns
        -------

        """
        time_series_name = df[self.groupby_col].iloc[0]
        # missing_tgt_vals是一个字典，key为空target值所对应的groupby col，value为这个groupby col 空值对应的_age
        if time_series_name in self._missing_tgt_vals.keys():
            # age_list相当于对age也做一个lag shift，和target同步
            age_list = full_df_ages.reindex([i - 1 for i in df.index.values.tolist()])
            # 前面的if代表如果missing tgts values 的字典里，时间序列的age和当前的age list有重复的话，说明有missing
            # targets的age需要填充， 那么就需要采样
            if not set(self._missing_tgt_vals[time_series_name]).isdisjoint(age_list) and df.shape[0] == window_size:
                # droplist [groupby, target]
                drop_list = [col for col in df.columns if
                             col == self.groupby_col or col == self.date_col or col == 'target']
                cont = tf.constant(
                    np.repeat(
                        df.drop(columns=self.cont_feats).values.reshape(1, window_size, -1),
                        batch_size, axis=0),
                    dtype=tf.float32)
                cat = tf.constant(
                    np.repeat(df[self.cat_feats + self.static_real_feats].values.reshape(1, window_size, -1),
                              batch_size, axis=0),
                    dtype=tf.float32)
                preds = model([cont, cat], training=training)[0][0]
                refill_indices = df.index[age_list.isin(self._missing_tgt_vals[time_series_name])]
                if df.index[0] > 0:
                    refill_values = [preds[i] for i in [r - df.index[0] for r in refill_indices]]
                else:
                    refill_values = [preds[i] for i in refill_indices]
                for idx, val in zip(refill_indices, refill_values):
                    df['prev_target'][idx] = val.numpy()[0]
        return df

    @property
    def max_age(self):
        return self._train_set_ages.max()

    @property
    def count_data(self):
        return self._count_data

    @property
    def mask_value(self):
        return self._mask_value

    @property
    def target_means(self):
        return self._target_means

    @property
    def train_set_ages(self):
        return self._train_set_ages

    @property
    def standard_scaler(self):
        return self._standard_scaler

    @property
    def raw_df(self):
        return self._raw_df.copy()

class TSTestDataset(TSTrainDataset):
    """
    输入的数据集不需要包含target
    """

    def __init__(self, ts_train: TSTrainDataset, test_df):
        self.ts_train = ts_train
        raw_df = self.ts_train.raw_df
        # 固定一下行列顺序
        self.df = test_df[raw_df.columns.drop('target')]
        self.test_len = len(test_df)
        self.inherit_train_ds()
        self._process_new_test_data()
        self._batch_test_data_prepared = False
        self._batch_idx = 0

    def inherit_train_ds(self):
        self.groupby_col = self.ts_train.groupby_col
        if self.groupby_col == 'category':
            self.df['category'] = 'dummy_cat'
        self.date_col = self.ts_train.date_col
        self.target_col = None
        self.freq = self.ts_train.freq
        self.cont_feats = self.ts_train.cont_feats
        self.cat_feats = self.ts_train.cat_feats
        self.static_real_feats = self.ts_train.static_real_feats
        self._count_data = self.ts_train.count_data
        self._label_encoder = self.ts_train._label_encoder
        self._standard_scaler = self.ts_train.standard_scaler
        self._target_means = self.ts_train.target_means
        self._encoder_dict = self.ts_train._encoder_dict
        self.lags = self.ts_train.lags

    def _process_new_test_data(self):
        """
        处理得到的test df
        """
        # 拼接train, test，顺便检查一下数据的连续性，然后做lag
        train_df = self.ts_train.raw_df
        train_df['ds_label'] = 'train'
        test_df = self.df.copy()
        test_df['ds_label'] = 'test'
        test_df['target'] = 0
        # temp_whole_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        # self._data_check(temp_whole_df)
        self.df = self._data_check(self.df)
        # self.df = self._add_lag_features(temp_whole_df)
        self._add_age_features()
        self._add_date_features()
        # 选取一样的特征，但是没有经过scale
        self._clean_features()
        self._set_forecast_horizon()

    def _add_lag_features(self, df):
        if self.lags is not None:
            self.lag_dict = {i: self.target_col for i in self.lags}
            groupby = df.groupby(self.groupby_col)
            for idx in self.lag_dict:
                df_add = groupby[self.lag_dict[idx]].shift(idx)
                new_col_name = f"{self.target_col}_lag_{idx}"
                df[new_col_name] = df_add
                df[new_col_name] = df[new_col_name].interpolate(limit_direction='both')
                # self.cont_feats.append(new_col_name)
        test_df = df[df['ds_label'] == 'test']
        assert len(test_df) == len(self.df)
        return test_df.drop(columns=['ds_label'])

    def _add_age_features(self):
        self._test_groups = self.df[self.groupby_col].unique()
        self._new_test_groups = []
        for group in self._test_groups:
            if group not in self.ts_train.ts_unique_values:
                # 新的时间序列
                self.ts_train.train_set_ages[group] = 0
                self._new_test_groups.append(group)
        self.df['_age'] = self.df.groupby(self.groupby_col).cumcount() + 1
        self.df['_age'] += self.ts_train.train_set_ages[self.df[self.groupby_col]].values

    def _clean_features(self):
        cur_cont_feats = [feat for feat in self.cont_feats if feat != 'prev_target']
        self.df = self.df[cur_cont_feats + self.cat_feats + self.static_real_feats]

    def _set_forecast_horizon(self):
        if len(self.df) > 0:
            self._horizon = self.df.groupby(self.groupby_col)['_age'].count().max()
        else:
            self._horizon = 0
        if self.lags and min(self.lags) > self._horizon:
            raise ParameterException('Lags can leak')

    def _standardize(self):
        """
        论文里说需要标准化到mean = 0， std = 1
        除了datetime字段，groupby字段和目标字段都要被标准化
        """
        covariate_names = [self.groupby_col] + list(self._encoder_dict.keys())
        covariate_mask = [False if col_name in covariate_names else True for col_name in self.df.columns]
        self.df.loc[:, covariate_mask] = self._standard_scaler.transform(self.df.loc[:, covariate_mask].astype('float'))

    def _standardize_df(self, df):
        covariate_names = [self.groupby_col] + list(self._encoder_dict.keys())
        covariate_mask = [False if col_name in covariate_names else True for col_name in self.df.columns]
        df.loc[:, covariate_mask] = self._standard_scaler.transform(
            df.loc[:, covariate_mask][self._standard_scaler.feature_names_in_].astype('float'))
        return df

    def next_batch(self, model, batch_size, window_size, include_all_training=False):
        """
        每次拿到下一个batch的训练集数据，为generator的辅助函数

        构建batch的过程是
        1. 根据groupby col，随机抽取batch size条时间序列
        2. 遍历这些时间序列 构建
        Parameters
        ----------
        model
        batch_size
        window_size
        valid_set: 如果True的话会在valid data里面去采样出batch
        kwargs

        Returns
        -------

        """
        if not self._batch_test_data_prepared:
            self._prepare_batched_test_data(batch_size, window_size, include_all_training)

        if self._batch_idx == self._horizon + self._train_batch_count:
            self._iterations += 1
            self._batch_idx = 0

        # return None if no more iterations
        if self._iterations >= self._total_iterations and self._iterations > 0:
            return (None, None, None, None)

        # 如果没有意外情况，得到next batch的数据
        # df_start_idx和df_end_idx描述的是在prepped_data_list中不同序列的获取范围
        df_start_idx = self._iterations * batch_size
        df_end_idx = (self._iterations + 1) * batch_size

        # batch_start_idx 和batch_end_idx描述的是在某一个取到的数据集的行范围
        if self._batch_idx >= self._train_batch_count:
            batch_start_idx = self._train_batch_count * window_size + self._batch_idx - self._train_batch_count
        else:
            batch_start_idx = self._batch_idx * window_size
        batch_end_idx = batch_start_idx + window_size

        batch_data = [df.iloc[batch_start_idx: batch_end_idx, :]
                      for df in self._prepped_data_list[df_start_idx:df_end_idx]]

        # 不需要如果targets没有nan的话
        # 如果batch data的大小不到窗口大小，由最后一行填充至窗口大小
        batch_data = [
            b_data.append(pd.concat([b_data.iloc[-1:, :]] * (window_size - b_data.shape[0]), ignore_index=True))
            if window_size - b_data.shape[0] > 0 else b_data
            for b_data in batch_data
        ]
        batch_df = pd.concat(batch_data)
        self._batch_idx += 1
        x_cont = batch_df[self.cont_feats].values.reshape(len(batch_data), window_size, -1)
        print(self.cont_feats)
        print(x_cont.shape)
        if len(batch_data) < batch_size:
            x_cont = np.append(x_cont, [x_cont[0]] * (batch_size - len(batch_data)), axis=0)
        x_cont = tf.Variable(x_cont, dtype=tf.float32)
        x_cats = []
        for cat in self.cat_feats + self.static_real_feats:
            x_cat = batch_df[cat].values.reshape(len(batch_data), window_size)
            if len(batch_data) < batch_size:
                x_cat = np.append(x_cat, [x_cat[0]] * (batch_size - len(batch_data)), axis=0)
            x_cats.append(tf.constant(x_cat, dtype=tf.float32))
        x_cat_keys = list(batch_df.groupby(self.groupby_col).groups.keys())
        x_scale_values = tf.constant(self.ts_train.target_means[x_cat_keys].values, dtype=tf.float32)

        return ([x_cont] + x_cats, x_scale_values, self._batch_idx - self._train_batch_count, self._iterations)

    def _prepare_batched_test_data(self, batch_size, window_size, include_all_training=False):
        """
        把测试集的数据打成各种时间窗口下的数据batch
        """
        if include_all_training:
            max_train_age = self.ts_train.df.groupby(self.ts_train.groupby_col).size().max()
        else:
            max_train_age = window_size

        self._train_batch_count = max_train_age // window_size
        # total iteration总共的时间序列数/batch +1, test_group是所有测试集里包括的时间序列
        self._total_iterations = len(self._test_groups) // batch_size + 1
        self._iterations = 0

        data_list = []
        for cat in self._test_groups:
            enc_cat = self.ts_train._label_encoder.transform([cat])[0]
            if enc_cat in self._new_test_groups:
                # 发现是一个全新时间序列
                logger.error(
                    f"There is new time series {self.groupby_col}={cat} not in training data but in test data!")
                # 这种情况其实需要填充一个全0的数据，暂时假设不会发生
                # train_data = pd
            else:
                # 如果是老时间序列，把对应的训练集数据拿出来
                train_data = self.ts_train.df[self.ts_train.df[self.groupby_col] == enc_cat]
                train_data = self._add_prev_target_col(train_data)
            if len(self.df) > 0:
                test_data = self.df[self.df[self.groupby_col] == cat]
                if cat in self._new_test_groups:
                    test_data[self.groupby_col] = 'dummy_test_category'
                test_data[self.cat_feats] = test_data[self.cat_feats].astype(str)
                for col in self._encoder_dict.keys():
                    test_data[col] = self._encoder_dict[col].transform(test_data[col])
                # test_data[self.groupby_col] = self._label_encoder.transform(test_data[self.groupby_col])
                test_data = self._standardize_df(test_data)
                test_data = self._add_prev_target_col(test_data, train_data)
                prepped_data = pd.concat([train_data.drop(columns=['target']), test_data]).reset_index(drop=True)
            else:
                prepped_data = train_data
            data_list.append(prepped_data)
        self._prepped_data_list = data_list
        self._batch_test_data_prepared = True
        self._batch_idx = 0

    def _add_prev_target_col(self, df, train_df=None):
        return super(TSTestDataset, self)._add_prev_target_col(df, train_df)

    @property
    def horizon(self):
        return self._horizon

    @property
    def test_groups(self):
        return self._test_groups

    @property
    def iterations(self):
        return self._iterations

    @property
    def batch_idx(self):
        return self._batch_idx

    @iterations.setter
    def iterations(self, value):
        self._iterations = value

    @batch_idx.setter
    def batch_idx(self, value):
        self._batch_idx = value

    @property
    def batch_test_data_prepared(self):
        return self._batch_test_data_prepared

    @batch_test_data_prepared.setter
    def batch_test_data_prepared(self, value):
        self._batch_test_data_prepared = value
