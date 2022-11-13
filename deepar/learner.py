import math
import pdb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, SimpleRNN, GRU, RNN, StackedRNNCells, \
    LSTMCell, GRUCell, SimpleRNNCell
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softplus
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError, Mean
import numpy as np
import pandas as pd
import logging
import os
from scipy import stats

from deepar.exceptions import ParameterException
from deepar.loss import GaussianLogLikelihood, negative_binomial_sampling, NegativeBinomialLogLikelihood
from deepar.ts_dataset import TSTrainDataset, TSTestDataset
from deepar.ts_generator import train_ts_generator, test_ts_generator
from deepar.callbacks import EarlyStopping
from deepar.utils import build_tf_lookup_hashtable, unscale

logger = logging.getLogger(__name__)


class DeepARLearner:
    def __init__(self, train_dataset: TSTrainDataset, cell_type='lstm', emb_dim=128, num_cells=128, num_layers=2,
                 learning_rate=0.001,
                 batch_size=64,
                 train_window=20, dropout=0.1, optimizer='adam', verbose=0, random_seed=None):
        """

        Parameters
        ----------
        train_dataset：训练集的TSTrainDataset实例
        cell_type: 中间层使用lstm，rnn还是gru
        emb_dim：类别类特征要进行embedding，那么embedding的output dim
        num_cells：每一层选定cell的个数
        num_layers：recurrent layer的层数
        learning_rate：学习率
        batch_size：batch大小
        train_window：采样窗口（回看窗口）的长度
        dropout：cell的dropout大小
        optimizer：优化器，'sgd'或者'adam'
        verbose：是否打出log
        random_seed：随机种子
        """
        if random_seed is not None:
            tf.random.set_seed(random_seed)
        if verbose == 1:
            logger.setLevel(logging.INFO)
        self._verbose = verbose
        self.cell_type = cell_type
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._dropout = dropout
        self.train_dataset = train_dataset
        self.emb_dim = emb_dim
        self.num_cells = num_cells
        self.num_layers = num_layers

        self.checkpoint_filepath = None
        # 如果train window_size 大于 目前时间序列的length，那么会被减少到时间序列的max length 也就是age
        if train_window > train_dataset.max_age:
            self._train_window = train_dataset.max_age
        self._train_window = train_window

        self.model = self._create_model()
        self._set_optimizer(optimizer)
        self._set_loss_function()

    def _create_model(self):
        num_cats = len(self.train_dataset.cat_feats) + len(self.train_dataset.static_real_feats)
        num_conts = len(self.train_dataset.cont_feats) + 1  # add one for prev_target
        print(f"number of cats : {num_cats}, number of conts: {num_conts}")
        # 首先对cat inputs 做embedding 然后concat到cont input上
        # 这个操作和gluonts.torch.modules.feature.FeatureEmbedder中保持一致，也是gluonts中对feat_static_cat的特征的处理
        # 对于每一个category feature，分别根据其cardinalities进行embedding，但保持统一的embedding输出维度
        cats_emb = []
        cat_inputs = []
        for cat in self.train_dataset.cat_feats + self.train_dataset.static_real_feats:
            cat_input = Input(shape=(self._train_window,), batch_size=self._batch_size)
            cats_nunique = self.train_dataset.df[cat].nunique() + 1
            embedding = Embedding(cats_nunique, self.emb_dim)(cat_input)
            cats_emb.append(embedding)
            cat_inputs.append(cat_input)
        sub_embs = tf.concat(cats_emb, axis=2)

        cont_inputs = Input(shape=(self._train_window, num_conts), batch_size=self._batch_size)
        concatenate = Concatenate()([cont_inputs, sub_embs])
        # last_layer = concatenate
        # for layer in range(self.num_layers):
        #     lstms = LSTM(self.lstm_dim, return_sequences=True, stateful=True, dropout=self._dropout,
        #                  recurrent_dropout=self._dropout, unit_forget_bias=True, name=f'layer_{layer}')(last_layer)
        #     last_layer = lstms
        if self.cell_type == 'lstm':
            rnn_cells = [LSTMCell(units=self.num_cells, name=f'lstm_{layer}', dropout=self._dropout,
                                  recurrent_dropout=self._dropout, ) for layer in range(self.num_layers)]
        elif self.cell_type == 'gru':
            rnn_cells = [GRUCell(units=self.num_cells, name=f'gru_{layer}', dropout=self._dropout,
                                 recurrent_dropout=self._dropout, ) for layer in range(self.num_layers)]
        elif self.cell_type == 'rnn':
            rnn_cells = [SimpleRNNCell(units=self.num_cells, name=f'rnn_{layer}', dropout=self._dropout,
                                       recurrent_dropout=self._dropout, ) for layer in range(self.num_layers)]
        else:
            raise ParameterException('cell type not supported')
        stacked_lstm = StackedRNNCells(rnn_cells)
        last_layer = RNN(stacked_lstm,
                         stateful=True,
                         return_sequences=True,
                         return_state=False, name='rnn')(concatenate)

        mu = Dense(
            1,
            activation='linear',
            kernel_initializer="glorot_normal",
            bias_initializer="glorot_normal",
            name="mu",
        )(last_layer)

        sigma = Dense(
            1,
            activation='softplus',
            kernel_initializer="glorot_normal",
            bias_initializer="glorot_normal",
            name="sigma",
        )(last_layer)
        inputs = [cont_inputs] + cat_inputs
        model = Model(inputs=inputs, outputs=[mu, sigma])
        return model

    def _set_optimizer(self, optimizer):
        if optimizer == "adam":
            self._optimizer = Adam(learning_rate=self._learning_rate)
        elif optimizer == "sgd":
            self._optimizer = SGD(lr=self._learning_rate, momentum=0.1, nesterov=True)
        else:
            raise ValueError("Optimizer must be one of `adam` and `sgd`")

    def _set_loss_function(self):
        mask_value = self.train_dataset.mask_value
        if self.train_dataset.count_data:
            self._loss_fn = NegativeBinomialLogLikelihood(mask_value)
        else:
            self._loss_fn = GaussianLogLikelihood(mask_value)

    def fit(self, epochs=100, steps_per_epoch=50, early_stopping=True, stopping_patience=5, stopping_delta=1,
            checkpoint_dir=None, tensorboard=True, validation=False):
        """
        对DeepAR模型训练，训练的轮次是 steps per epoch * epochs
        Returns
        -------

        """
        self._epochs = epochs
        self._save_checkpoints(checkpoint_dir, tensorboard)
        train_gen = train_ts_generator(self.train_dataset, self.model, self._batch_size, self._train_window, False)
        if validation:
            val_gen = train_ts_generator(self.train_dataset, self.model, self._batch_size, self._train_window, True)
        else:
            val_gen = None
        # Iterate over epochs.
        return self._iterate_train_loop(
            train_gen,
            val_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            early_stopping=early_stopping,
            stopping_patience=stopping_patience,
            stopping_delta=stopping_delta, )

    def _save_checkpoints(self, checkpoint_dir, tensorboard):
        """
        如果第一次跑，创建对于的文件夹保存检查点
        如果并非第一次跑，直接读取最新的检查点
        Parameters
        ----------
        checkpoint_dir

        Returns
        -------

        """
        self._checkpointer = tf.train.Checkpoint(
            optimizer=self._optimizer, model=self.model
        )
        if checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.checkpoint_filepath = os.path.join(checkpoint_dir, "{epoch:04d}.ckpt")
            latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_ckpt:
                self._checkpointer.restore(latest_ckpt)
        elif tensorboard:
            # create tensorboard log files in default location
            checkpoint_dir = "./tb/"
            tb_writer = tf.summary.create_file_writer(checkpoint_dir)
            tb_writer.set_as_default()
        else:
            self.checkpoint_filepath = None
        self._tb = tensorboard

    def _iterate_train_loop(self, train_gen, val_gen, epochs, steps_per_epoch, early_stopping,
                            stopping_patience, stopping_delta):
        # 定义一个batch的loss， 整个epoch的loss 和 整个过程的loss
        batch_loss_avg = Mean()
        epoch_loss_avg = Mean()
        eval_loss_avg = Mean()
        eval_mae = MeanAbsoluteError()
        eval_rmse = RootMeanSquaredError()

        # 设置early_stopping
        early_stopping_cb = EarlyStopping(
            patience=stopping_patience, active=early_stopping, delta=stopping_delta
        )
        self.v_hashtable = build_tf_lookup_hashtable(self.train_dataset.target_means)
        best_metric = math.inf
        for epoch in range(epochs):
            logger.info(f"Start of epoch {epoch}")
            for batch, (x_batch_train, cat_labels, y_batch_train) in enumerate(train_gen):
                with tf.GradientTape(persistent=True) as tape:
                    mu, presigma = self.model(x_batch_train, training=True)
                    mu, sigma = self._softplus(mu, presigma)
                    mu, sigma = unscale(mu, sigma, cat_labels, self.v_hashtable)
                    loss_value = self._loss_fn(y_batch_train, (mu, sigma))
                # tensorboard
                if self._tb:
                    tf.summary.scalar('train_loss', loss_value, epoch * steps_per_epoch + batch)

                batch_loss_avg(loss_value)
                epoch_loss_avg(loss_value)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # Log 5x per epoch. 每一段batch输出一下
                if batch % (steps_per_epoch // 5) == 0 and batch != 0:
                    logger.info(
                        f"Epoch {epoch}: Avg train loss over last {(steps_per_epoch // 5)} steps: {batch_loss_avg.result()}"
                    )
                    batch_loss_avg.reset_states()

                #
                epoch_loss_avg_result = epoch_loss_avg.result()
                if batch == steps_per_epoch:
                    logger.info(f"Epoch {epoch} over {steps_per_epoch}: Avg train loss {epoch_loss_avg_result}")
                    break
            # validation
            if val_gen is not None:
                logger.info(f"End of epoch {epoch}, validating...")

                for batch, (x_batch_val, cat_labels, y_batch_val) in enumerate(val_gen):
                    with tf.GradientTape() as tape:
                        mu, sigma = self.model(x_batch_val, training=True)
                        mu, sigma = self._softplus(mu, presigma)
                        mu, sigma = unscale(mu, sigma, cat_labels, self.v_hashtable)
                        loss_value = self._loss_fn(y_batch_val, (mu, sigma))

                    eval_mae(y_batch_val, mu)
                    eval_rmse(y_batch_val, mu)
                    eval_loss_avg(loss_value)
                    if batch == steps_per_epoch:
                        break
                # logging
                logger.info(f"Epoch {epoch}: Val loss on {steps_per_epoch} steps: {eval_loss_avg.result()}")
                logger.info(f"Epoch {epoch}: Val MAE: {eval_mae.result()}, RMSE: {eval_rmse.result()}")
                if self._tb:
                    tf.summary.scalar("val_loss", eval_loss_avg.result(), epoch)
                    tf.summary.scalar("val_mae", eval_mae.result(), epoch)
                    tf.summary.scalar("val_rmse", eval_rmse.result(), epoch)
                new_metric = eval_mae.result()

                if early_stopping_cb(eval_mae.result()):
                    break
                # reset metric states
                eval_loss_avg.reset_states()
                eval_mae.reset_states()
                eval_rmse.reset_states()
            else:
                # 没有验证集 通过训练集分数来看earlystopping
                if early_stopping_cb(epoch_loss_avg_result):
                    break
                new_metric = epoch_loss_avg_result

            # update best_metric and save new checkpoint if improvement
            if new_metric < best_metric:
                best_metric = new_metric
                if self.checkpoint_filepath is not None:
                    self._checkpointer.save(file_prefix=self.checkpoint_filepath)
                else:
                    self.model.save_weights("model_best_weights.h5")
            epoch_loss_avg.reset_states

        if self.checkpoint_filepath is None:
            self.model.load_weights("model_best_weights.h5")
            os.remove("model_best_weights.h5")
        return best_metric, epoch + 1

    def _softplus(self, mu, presigma):
        sigma = softplus(presigma)
        # 如果是count data 负二项分布，mu也要是正数
        if self.train_dataset.count_data:
            mu = softplus(mu)
        return mu, sigma

    def predict(self, test_dataset: TSTestDataset, horizon=None, samples=100, point_estimate=False,
                confidence_interval=False,
                confidence_level=0.95, include_all_training=False, return_in_sample_predictions=True):
        """

        Parameters
        ----------
        test_dataset
        horizon
        samples
        point_estimate
        confidence_interval
        confidence_level
        include_all_training
        return_in_sample_predictions

        Returns
        -------
        predictions, shape is (# unique test groups, horizon, # samples)

        """
        test_gen = test_ts_generator(test_dataset, self.model, self._batch_size, self._train_window,
                                     include_all_training)
        # for layer in range(self.num_layers):
        #     self.model.get_layer(f'layer_{layer}').reset_states()
        self.model.get_layer('rnn').reset_states()
        if horizon is None:
            logger.info(f"set horizon to test dataset horizon: {test_dataset.horizon}")
            horizon = test_dataset.horizon
        elif horizon > test_dataset.horizon:
            logger.error(f"input horizon is not legal")

        test_samples = [[] for _ in range(len(test_dataset.test_groups))]
        prev_iteration_index = 0

        for batch_idx, batch in enumerate(test_gen):
            x_test, scale_keys, horizon_idx, iteration_index = batch
            if iteration_index is None:
                break
            if horizon_idx == horizon:
                test_dataset.batch_idx = 0
                test_dataset.iterations += 1
                continue

            # reset lstm states for new sequence of predictions through time
            if iteration_index != prev_iteration_index:
                self._model.get_layer("rnn").reset_states()
            print(f"horizon idx :{horizon_idx}")
            # 祖先采样
            if horizon_idx > 1:
                x_test_new = x_test[0][:, :1, -1:].assign(mu[:, :1, :])
                x_test = [x_test_new] + x_test[1:]
            # make predictions
            # import pdb
            # pdb.set_trace()
            mu, presigma = self.model(x_test)
            mu, sigma = self._softplus(mu, presigma)
            # unscale
            scaled_mu, scaled_sigma = unscale(
                mu[: scale_keys.shape[0]],
                sigma[: scale_keys.shape[0]],
                scale_keys,
                self.v_hashtable,
            )
            # in-sample predictions (ancestral sampling)
            if horizon_idx <= 0:
                print('in-sample ancestral sampling')
                if horizon_idx % 5 == 0:
                    logger.info(
                        f"Making in-sample predictions with ancestral sampling. {-horizon_idx} batches remaining"
                    )
                scaled_mu, scaled_sigma = self._squeeze(scaled_mu, scaled_sigma)

                for mu_sample, sigma_sample, sample_list in zip(scaled_mu, scaled_sigma,
                                                                test_samples[iteration_index * self._batch_size: (
                                                                                                                         iteration_index + 1) * self._batch_size], ):
                    draws = self._draw_samples(mu_sample, sigma_sample, point_estimate=point_estimate, samples=samples,
                                               confidence_interval=confidence_interval,
                                               confidence_level=confidence_level)
                    sample_list.extend(draws)
            # draw samples from learned distributions for test samples
            else:
                print('learn from test samples')
                if horizon_idx % 5 == 0:
                    logger.info(
                        f"Making future predictions. {horizon - horizon_idx} batches remaining"
                    )
                scaled_mu, scaled_sigma = scaled_mu[:, :1, :], scaled_sigma[:, :1, :]
                squeezed_mu, squeezed_sigma = self._squeeze(scaled_mu, scaled_sigma, squeeze_dims=[1, 2])
                draws = self._draw_samples(squeezed_mu, squeezed_sigma, point_estimate=point_estimate, samples=samples,
                                           confidence_interval=confidence_interval,
                                           confidence_level=confidence_level)
                for draw_list, sample_list in zip(draws, test_samples[iteration_index * self._batch_size: (
                                                                                                                  iteration_index + 1) * self._batch_size], ):
                    sample_list.append(draw_list)
        test_dataset.batch_idx = 0
        test_dataset.iterations = 0
        test_dataset.batch_test_data_prepared = False
        import pdb
        pdb.set_trace()
        if return_in_sample_predictions:
            pred_samples = np.array(test_samples)[:, -(self.train_dataset.max_age + horizon):, :]
        else:
            pred_samples = np.array(test_samples)[:, -horizon:, :]
        return pred_samples

    def _squeeze(self, mu, sigma, squeeze_dims=[2]):
        return tf.squeeze(mu, squeeze_dims), tf.squeeze(sigma, squeeze_dims)

    def _draw_samples(self, mu_tensor, sigma_tensor, point_estimate=False, confidence_interval=False,
                      confidence_level=0.95, samples=1):
        """
        采样
        如果是点预测 直接返回mu
        如果是负二项分布 按照负二项分布进行采样
        如果是高斯分布 按照高斯分布进行采样

        """
        # TODO: 支持分位数 np.quantile
        if point_estimate:
            return [np.repeat(mu, samples) for mu in mu_tensor]
        if confidence_interval and confidence_level < 1 and confidence_level > 0:
            z_score = stats.norm.ppf(confidence_level)
            import pdb
            pdb.set_trace()
            return [[(mu - sigma * z_score).numpy(), mu.numpy(), (mu + sigma * z_score).numpy()] for mu, sigma in
                    zip(mu_tensor, sigma_tensor)]
        elif self.train_dataset.count_data:
            import pdb
            pdb.set_trace()
            return [list(negative_binomial_sampling(mu, sigma, samples)) for mu, sigma in zip(mu_tensor, sigma_tensor)]
        else:
            return [list(np.random.normal(mu, sigma, samples)) for mu, sigma in zip(mu_tensor, sigma_tensor)]
