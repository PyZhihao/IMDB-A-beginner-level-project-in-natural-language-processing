# IMDB Movie Review Sentiment Analysis

这是一个用于学习 NLP 基础流程的小项目，任务是对 IMDB 电影评论进行二分类情感分析：

- `0`：负面评论
- `1`：正面评论

项目使用 PyTorch 从数据读取、分词、构建词表、Dataset/DataLoader、模型训练到测试评估完整实现了一遍，并分别对比了 RNN、LSTM 和 GRU 三种循环神经网络结构。

## 项目结构

```text
.
├── data.py               # 数据读取、分词、Dataset
├── vocab.py              # Vocabulary
├── utils.py              # seed、train、evaluate、checkpoint
├── models/
│   ├── __init__.py
│   ├── rnn.py
│   ├── gru.py
│   └── lstm.py
├── main_rnn.py           # RNN 训练入口
├── main_gru.py           # GRU 训练入口
├── main_lstm.py          # LSTM 训练入口
├── RNN.md                # RNN 原理笔记
├── GRU.md                # GRU 原理笔记
├── LSTM.md               # LSTM 原理笔记
├── README.md
├── requirements.txt
└── .gitignore
```

## 数据准备

本项目使用 Stanford IMDB Large Movie Review Dataset。

下载地址：

```text
https://ai.stanford.edu/~amaas/data/sentiment/
```

下载并解压后，将 `aclImdb` 文件夹放到项目根目录下，目录结构应类似：

```text
aclImdb/
├── train/
│   ├── neg/
│   └── pos/
└── test/
    ├── neg/
    └── pos/
```

数据规模：

- 训练集：25000 条评论
- 测试集：25000 条评论
- 每个集合中正负样本各占一半

## 数据处理流程

代码中的主要处理步骤如下：

1. 读取 `train/neg`、`train/pos`、`test/neg`、`test/pos` 下的文本文件。
2. 将评论文本转为小写。
3. 使用正则表达式进行简单英文分词，支持单词、数字、标点以及 `don't`、`i'm` 这类缩写。
4. 将原始训练集按 `80% / 20%` 划分为训练集和验证集。
5. 只使用训练集构建词表，避免验证集和测试集信息泄漏。
6. 词表包含两个特殊 token：
   - `<pad>`：索引为 `0`
   - `<unk>`：索引为 `1`
7. 每条评论最多保留 `256` 个 token，超过则截断，不足则 padding。
8. 使用真实序列长度配合 `pack_padded_sequence`，让模型忽略 padding 部分。

## 模型

三个模型的整体结构基本一致：

```text
Token IDs
   ↓
Embedding
   ↓
RNN / LSTM / GRU
   ↓
最后一层最终隐藏状态
   ↓
Dropout
   ↓
Linear
   ↓
情感分类 logit
```

当前主要超参数：

| 参数 | 值 |
| --- | --- |
| batch size | 64 |
| epochs | 10 |
| learning rate | 1e-3 |
| max sequence length | 256 |
| max vocab size | 20000 |
| embedding dim | 128 |
| hidden dim | 128 |
| dropout | 0.3 |

训练使用：

- 损失函数：`BCEWithLogitsLoss`
- 优化器：`Adam`
- 梯度裁剪：`clip_grad_norm_(max_norm=1.0)`
- 最优模型选择：根据验证集 `val_loss` 最低保存模型

## 运行方式

安装 PyTorch 后，在项目根目录运行：

```bash
python main_rnn.py
python main_lstm.py
python main_gru.py
```

每个脚本都会完成以下流程：

1. 读取数据
2. 构建词表
3. 训练模型
4. 根据最低验证集 loss 保存最优模型
5. 加载最优模型
6. 在测试集上输出最终结果

## 实验结果

训练过程中会输出每个 epoch 的训练集 loss、训练集准确率、验证集 loss 和验证集准确率。训练结束后，根据最低验证集 loss 加载最优模型，并在测试集上评估最终结果。

### RNN

| 模型 | Min val loss | Best Val Acc | Test Loss | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| RNN 1 layer | 0.5486 | 0.7630 | **0.5699** | **0.7583** |
| RNN 2 layers | 0.5781 | 0.7466 | 0.5871 | 0.7437 |
| RNN 3 layers | 0.5572 | 0.7502 | 0.5756 | 0.7364 |

RNN 1 层效果最好，测试准确率 `0.7583`。增加到 2 层和 3 层后，验证集和测试集表现没有提升，反而下降。

普通 RNN 对长序列建模能力有限，虽然加深层数理论上可以增强表达能力，但也会带来训练难度增加、梯度传播不稳定、过拟合风险上升等问题。在 IMDB 这种较长文本任务中，普通 RNN 难以有效保留长距离依赖信息，因此增加层数并没有带来收益。

### LSTM

| 模型 | Min val loss | Best Val Acc | Test Loss | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| LSTM 1 layer | 0.4060 | 0.8364 | 0.4208 | 0.8314 |
| LSTM 2 layers | 0.3921 | 0.8368 | 0.4027 | 0.8320 |
| LSTM 3 layers | 0.3842 | 0.8368 | **0.3889** | **0.8334** |

LSTM 从 1 层到 3 层，`val_loss` 和 `test_loss` 持续下降，测试准确率也从 `0.8314` 提升到 `0.8334`，但提升幅度比较小。

LSTM 通过细胞状态和门控机制缓解了普通 RNN 的梯度消失问题，能够更好地保留长距离信息。增加层数可以提升模型表达能力，但在当前数据规模和超参数设置下，1 层 LSTM 已经能学到较强特征，因此 2 层和 3 层只带来有限提升。

### GRU

| 模型 | Min val loss | Best Val Acc | Test Loss | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| GRU 1 layer | 0.3690 | 0.8614 | 0.3809 | 0.8509 |
| GRU 2 layers | 0.3236 | 0.8596 | **0.3349** | 0.8516 |
| GRU 3 layers | 0.3247 | 0.8664 | 0.3384 | **0.8587** |

GRU 整体表现最好。1 层 GRU 已经达到 `0.8509` 的测试准确率；2 层 GRU 取得最低测试损失 `0.3349`；3 层 GRU 取得最高测试准确率 `0.8587`。

GRU 使用更新门和重置门控制信息流动，相比普通 RNN 能更好处理长序列依赖；相比 LSTM，GRU 结构更简单、参数更少，训练效率更高。在本项目中，3 层 GRU 的分类准确率最高，但 2 层 GRU 的测试损失最低，说明 2 层和 3 层 GRU 都是较优选择。

### 最终结论

RNN 结构简单，但长序列中容易丢失早期信息。

LSTM 通过输入门、遗忘门、输出门和细胞状态增强了长期依赖建模能力。

GRU 用更简洁的门控结构实现了类似 LSTM 的效果，在本实验中取得了更好的性能和泛化表现。
