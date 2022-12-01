# README

# Overview

代码遵循DeepAR原始论文，使用TensorFlow2的框架建立

代码API，框架结构参考：

[https://github.com/awslabs/gluonts/](https://github.com/awslabs/gluonts/) （API命名尽量相同）

[https://github.com/jdb78/pytorch-forecasting](https://github.com/jdb78/pytorch-forecasting) （Pytorch-forecasting实现）

[https://github.com/arrigonialberto86/deepar](https://github.com/arrigonialberto86/deepar) （貌似是paperwithcode上唯一的TF实现，主要代码框架基于，甚至一些函数的内容都基于这个）

[https://github.com/husnejahan/DeepAR-pytorch](https://github.com/husnejahan/DeepAR-pytorch) （线下写的比较好的Pytorch实现）

# 整体调用API

```python
train_ds = TSTrainDataset(df, xxx, xxx)
test_ds = TSTestDataset(test_df, ...)
learner = DeepARLearner(ts_ds)
learner.fit()
pred_df = learner.predict(test_ds)
```

# 数据集层

在ts_dataset.py中，实现了一个抽象类和两个实现类。

两个实现类分别是`TSTrainDataset 和``TSTestDataset`

TSTrainDataset会输入数据集和数据集相关参数，数据集处理的相关参数，依次完成如下的数据集处理：

1. 数据基本检查，目的是检查输入的数据集是否符合要求：
    1. 主键空值
    2. 主键唯一性
    3. 目标列数值
    4. 数据日期连续性
    5. 日期和输入的freq的一致性
    
    其中如果日期连续性和freq一致性出现问题，在utils.py中提供了相关的函数进行预处理`date_filler, date_aggregator`
    
2. 增加日期相关特征
3. 增加age特征（依据原始论文）
4. 对类别型特征进行label encoder
5. 特征标准化

TSTestDataset中除了输入测试集之外，还需要对应的训练数据的TSTrainDataset的实例，用来辅助将训练的时候的变量传递，例如放缩系数v，测试集的age特征要基于训练集等。

# 模型层：

## layers

基础DeepAR网络结构层，学习比如高斯分布的mu和sigma的基础网络结构`GaussianLayer`

## learner

基础的DeepAR的封装，主要是fit,predict两个方法

根据DeepAR原始论文和Gluonts的代码，其中论文中并没有特别说明特征的使用情况。Gluonts的src/gluonts/mx/model/deepar/_estimator.py下的代码可以看到，`FEAT_STATIC_REAL`和`FEAT_STATIC_CAT` 分别就做了AsNumpy，Embedding的处理，`FEAT_DYNAMIC_REAL` 和生成的age和time_features(day, week, month这类)一起Vstack起来，最后把embedding完的部分和vstack后的部分连接进入模型。

```python
可以通过DeepARLearner(ds).model.summary()看到模型的情况
```

fit方法就是正常的根据epoch/iteration进行训练，使用的是TensorFlow2的eager execution模式

predict方法除了对训练集计算出mu,sigma，还要对数据horizon不为1的结果进行祖先采样（依据原始论文），还原放缩变换，并对输出结果进行采样。

## loss

遵循原始论文，有高斯分布和负二项分布，除了根据公式计算loss外，还增加了mask，如果遇到missing value不应该计入loss的计算

## ts_generator

生成对应的batch的生成器，整个代码的思路是`TSTrainDataset` 初始化的时候完成数据的清洗，处理，然后在调用生成器的时候把它根据train_window打成符合Learner模型输入的窗口化的数据，然后并输入。
