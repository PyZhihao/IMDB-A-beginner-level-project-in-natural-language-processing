# LSTM（长短期记忆网络）

## 1. LSTM 是什么

`LSTM`，全称： $\text{Long Short-Term Memory}$，中文叫做：长短期记忆网络。

`LSTM` 是一种用于处理**序列数据**的循环神经网络结构，是对传统 `RNN` 的改进。

与 `RNN` 类似，`LSTM` 也通过隐藏状态 $h_t$ 将之前的信息传递到当前时间步。

除此之外，`LSTM` 还额外引入了一个细胞状态 $C_t$ 用于保存和传递长期信息。

因此，`LSTM` 在时间步之间会同时传递：

- 细胞状态（`Cell State`）： $C_t$
- 隐藏状态（`Hidden State`）： $h_t$

`LSTM` 通过三个门控机制控制信息的流动：

- 遗忘门（`Forget Gate`）
- 输入门（`Input Gate`）
- 输出门（`Output Gate`）

---

## 2. 为什么需要 LSTM

传统 `RNN` 的隐藏状态更新为： $h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)$

每一个时间步都会直接根据 $x_t$ 和 $h_{t-1}$ 重新计算隐藏状态。

在长序列中，随着时间步不断增加，早期的重要信息可能逐渐丢失，并且在反向传播过程中容易出现梯度消失问题。

也就是说，传统 `RNN` **没有专门的机制决定哪些历史信息应该长期保留、哪些信息应该遗忘。**

`LSTM` 的核心思想就是：**引入一个专门用于保存长期信息的细胞状态 $C_t$，并通过遗忘门、输入门和输出门控制信息的保留、写入和输出。**

---

## 3. LSTM 的核心结构

`LSTM` 主要包含：

1. 遗忘门（`Forget Gate`）
2. 输入门（`Input Gate`）
3. 候选细胞状态（`Candidate Cell State`）
4. 细胞状态（`Cell State`）
5. 输出门（`Output Gate`）
6. 隐藏状态（`Hidden State`）

核心结构：

1. 遗忘门： $f_t=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+b_f)$

2. 输入门： $i_t=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_i)$

3. 候选细胞状态： $\tilde{C}_t=\tanh(W_{xc}x_t+W_{hc}h_{t-1}+b_c)$

4. 当前细胞状态： $C_t=f_t\odot C_{t-1}+i_t\odot\tilde{C}_t$

5. 输出门： $o_t=\sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_o)$

6. 当前隐藏状态： $h_t=o_t\odot\tanh(C_t)$

---

## 4. 遗忘门（Forget Gate）

遗忘门： $\boxed{f_t=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+b_f)}$

其中：

- $x_t$：当前时间步输入
- $h_{t-1}$：上一时间步隐藏状态
- $W_{xf}$、 $W_{hf}$：可学习权重参数
- $b_f$：可学习偏置参数
- $\sigma$：`Sigmoid` 激活函数

`Sigmoid` 的输出范围为： $0<f_t<1$

遗忘门的主要作用：**控制保留多少上一时间步细胞状态 $C_{t-1}$ 信息进入当前时间步细胞状态 $C_{t}$**

---

## 5. 输入门（Input Gate）

输入门： $\boxed{i_t=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_i)}$

其中：

- $x_t$：当前时间步输入
- $h_{t-1}$：上一时间步隐藏状态
- $W_{xi}$、 $W_{hi}$：可学习权重参数
- $b_i$：可学习偏置参数
- $\sigma$：`Sigmoid` 激活函数

同样： $0<i_t<1$

输入门的主要作用：**控制保留多少当前时间步候选细胞状态 $\tilde{C}_t$ 信息进入当前时间步细胞状态 $C_{t}$**

---

## 6. 候选细胞状态

候选细胞状态为： $\boxed{\tilde{C}_t=\tanh(W_{xc}x_t+W_{hc}h_{t-1}+b_c)}$

其中：

- $x_t$：当前输入
- $h_{t-1}$：上一时间步隐藏状态
- $W_{xc}$、 $W_{hc}$：可学习权重参数
- $b_c$：可学习偏置参数
- $\tanh$：双曲正切激活函数

`tanh` 的输出范围为： $-1<\tilde{C}_t<1$

候选细胞状态 $\tilde{C}_t$，可以理解为：**根据当前时间步输入 $x_{t}$ 和上一时间步隐藏状态 $h_{t-1}$ 得到的候选信息**

---

## 7. 当前细胞状态

得到遗忘门、输入门和候选细胞状态之后，需要更新当前时间步细胞状态： $\boxed{C_t=f_t\odot C_{t-1}+i_t\odot\tilde{C}_t}$

**保留旧信息：** $f_t\odot C_{t-1}$

**加入新信息：** $i_t\odot\tilde{C}_t$

细胞状态 $C_t$ 是 `LSTM` 中用于保存和传递长期信息的核心状态。

---

## 8. 输出门（Output Gate）

输出门： $\boxed{o_t=\sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_o)}$

其中：

- $x_t$：当前时间步输入
- $h_{t-1}$：上一时间步隐藏状态
- $W_{xo}$、 $W_{ho}$：可学习权重参数
- $b_o$：可学习偏置参数
- $\sigma$：`Sigmoid` 激活函数

同样： $0<o_t<1$

输出门主要作用：**控制保留多少当前时间步细胞状态 $C_t$ 信息进入当前隐藏状态 $h_t$。**

## 9. 当前隐藏状态

得到候选细胞状态之后，需要计算当前隐藏状态： $\boxed{h_t=o_t\odot\tanh(C_t)}$

---

## 9. 如何理解三个门

遗忘门： $f_t$，用于计算当前时间步细胞状态： $C_t=f_t\odot C_{t-1}+i_t\odot\tilde{C}_t$

**遗忘门决定的是上一时间步细胞状态中的旧信息中，有多少进入当前时间步细胞状态**

当 $f_t\approx0$，此时： $f_t\odot C_{t-1}\approx0$，说明对应的旧信息基本被遗忘。

当 $f_t\approx1$，此时： $f_t\odot C_{t-1}\approx C_{t-1}$，说明对应的旧信息基本被保留。



输入门： $i_t$，用于计算当前时间步细胞状态： $C_t=f_t\odot C_{t-1}+i_t\odot\tilde{C}_t$

**输入门决定的是当前时间步产生的新候选细胞信息中，有多少进入当前时间步细胞状态**

当 $i_t\approx0$，此时： $i_t\odot\tilde{C}_t\approx0$，说明基本不写入当前的新信息。

当 $i_t\approx1$，此时： $i_t\odot\tilde{C}_t\approx1$，说明当前候选新信息基本全部写入。



输出门： $o_t$，用于计算当前时间步的隐藏状态： $h_t=o_t\odot\tanh(C_t)$

**输出门决定的是当前时间步细胞状态的信息中，有多少输出为当前时间步隐藏状态**

当 $o_t\approx0$，此时： $o_t\odot\tanh(C_t)\approx0$，说明当前细胞状态中的信息基本不输出。

当 $o_t\approx1$，此时： $o_t\odot\tanh(C_t)\approx1$，说明当前细胞状态中的信息基本全部参与隐藏状态的生成。

---

## 10. LSTM 一个时间步的完整计算

给定当前时间步输入 $x_t$，上一时间步隐藏状态 $h_{t-1}$，上一时间步细胞状态 $C_{t-1}$

第一步：计算遗忘门： $f_t=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+b_f)$

第二步：计算输入门： $i_t=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_i)$

第三步：计算候选细胞状态： $\tilde{C}_t=\tanh(W_{xc}x_t+W_{hc}h_{t-1}+b_c)$

第四步：更新当前细胞状态： $C_t=f_t\odot C_{t-1}+i_t\odot\tilde{C}_t$

第五步：计算输出门： $o_t=\sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_o)$

第六步：计算当前隐藏状态： $h_t=o_t\odot\tanh(C_t)$

---

## 11. LSTM 如何处理整个序列

假设输入序列： $x_1,x_2,x_3,\cdots,x_T$

首先初始化： $h_0$ 和 $C_0$，一般初始化为全 $0$。

然后随时间步进行状态更新

```text
x1 → LSTM → h1, C1
              ↓
x2 → LSTM → h2, C2
              ↓
x3 → LSTM → h3, C3
              ↓
             ...
              ↓
xT → LSTM → hT, CT
```

每一个时间步内部都会计算：

* 遗忘门 $f_t$
* 输入门 $i_t$
* 候选细胞状态 $\tilde{C}_t$
* 当前细胞状态 $C_t$
* 输出门 $o_t$
* 当前隐藏状态 $h_t$

---

## 12. LSTM 的参数共享

**LSTM 在所有时间步共享同一套参数。**

例如：

$$
W_{xf},W_{hf},b_f
$$

$$
W_{xi},W_{hi},b_i
$$

$$
W_{xc},W_{hc},b_c
$$

$$
W_{xo},W_{ho},b_o
$$

---

## 13. PyTorch 中的 LSTM

```python
self.lstm = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    num_layers=1,
    bias=True,
    batch_first=True,
    dropout=0.0,
    bidirectional=False
)
```

主要参数：

| **参数含义**    |                            |
| --------------- | -------------------------- |
| `input_size`    | 每个时间步输入特征的维度   |
| `hidden_size`   | 隐藏状态和细胞状态的维度   |
| `num_layers`    | LSTM 的层数                |
| `bias`          | 是否使用偏置               |
| `batch_first`   | 是否让 Batch 位于第一维    |
| `dropout`       | 多层 LSTM 层之间的 Dropout |
| `bidirectional` | 是否使用双向结构           |

---

## 14. LSTM 的输入 Shape

`LSTM` 的输入 `Shape` 与 `RNN`、`GRU` 一致。

##### 当设置 `batch_first=True`

输入 `x` 的 `Shape`： $[\text{batch\_size},\text{seq\_len},\text{input\_size}]$

例如：

```text
batch_size = 32
seq_len = 200
embedding_dim = 128
```

`x.shape` 为：

```text
[32, 200, 128]
```

含义：

* `32`：一个 `Batch` 中有 `32` 个样本
* `200`：每个序列有 `200` 个时间步（时间步大小）
* `128`：每个时间步输入向量维度为 `128`

---

##### 当设置 `batch_first=False`

输入 `x` 的 `Shape` 为： $[\text{seq\_len},\text{batch\_size},\text{input\_size}]$

例如：

```text
batch_size = 32
seq_len = 200
embedding_dim = 128
```

`x.shape` 为：

```text
[200, 32, 128]
```

含义：

* `200`：每个序列长度为 `200`，即时间步大小
* `32`：一个 `Batch` 中有 `32` 个样本
* `128`：每个时间步输入向量维度为 `128`

`batch_first` **只会交换前两个维度的位置，不会改变数据本身的含义。**

另外需要注意：

`batch_first` 只影响输入 `x` 和输出 `output` 的维度顺序，不影响 `h_n` 和 `c_n` 的维度顺序。

---

## 15. LSTM 的输出

调用：

```python
output, (h_n, c_n) = self.lstm(x)
```

会返回：

```text
output
h_n
c_n
```

---

### 15.1 `output`

当 `batch_first=True`，且为单向 `LSTM` 时，`output`的`shape`为： $[\text{batch\_size},\text{seq\_len},\text{hidden\_size}]$

例如：

```text
[32, 200, 128]
```

`output` 表示：**最后一层 LSTM 在所有时间步产生的隐藏状态。**

也就是： $[h_1,h_2,\cdots,h_T]$

例如：`output[:, 0, :]`表示最后一层第一个时间步的隐藏状态 $h_1$

`output[:, -1, :]`表示最后一层最后一个时间步的隐藏状态 $h_T$

---

### 15.2 `h_n`

`h_n` 的 `Shape` 为： $[\text{num\_layers},\text{batch\_size},\text{hidden\_size}]$

例如：

```text
[2, 32, 128]
```

`h_n` 表示：**每一层 LSTM 在最后一个时间步的隐藏状态。**

例如： $h_n[-1]$ 表示最后一层 `LSTM` 在最后一个时间步的隐藏状态，其 `Shape` 为：`[32, 128]`

---

### 15.3 `c_n`

`c_n` 的 `Shape` 为： $[\text{num\_layers},\text{batch\_size},\text{hidden\_size}]$

例如：

```text
[2, 32, 128]
```

`c_n` 表示：**每一层 LSTM 在最后一个时间步的细胞状态。**

例如：`c_n[-1]`表示最后一层 `LSTM` 在最后一个时间步的细胞状态，其 `Shape` 为：`[32, 128]`

---

### 15.4 `output`、`h_n` 和 `c_n`

可以简单记：

```text
output：最后一层 + 所有时间步的隐藏状态 h

h_n：所有层 + 最后时间步的隐藏状态 h

c_n：所有层 + 最后时间步的细胞状态 C
```

对于普通单向 `LSTM`：`output[:, -1, :]` 和 `h_n[-1]`通常表示同一个最终隐藏状态。

---

## 16. 多层 LSTM

如果`num_layers=2`，那么就是两层 `LSTM`。

第一层： $x_t\rightarrow h_t^{(1)}$

第二层接收第一层当前时间步的隐藏状态作为输入： $h_t^{(1)}\rightarrow h_t^{(2)}$

同时，每一层都会维护自己的： $h_t$ 和 $C_t$

```text
             时间方向 →

Layer 2:    h1² → h2² → h3² → h4²
             ↑     ↑     ↑     ↑
Layer 1:    h1¹ → h2¹ → h3¹ → h4¹
             ↑     ↑     ↑     ↑
            x1    x2    x3    x4
```

每一层内部还存在自己的细胞状态：

```text
Layer 2:    C1² → C2² → C3² → C4²

Layer 1:    C1¹ → C2¹ → C3¹ → C4¹
```

因此多层 `LSTM` 中有两个方向的信息传递：

* 横向：时间步之间传递隐藏状态 $h_t$ 和细胞状态 $C_t$
* 纵向：上一层的隐藏状态 $h_t$ 作为下一层当前时间步的输入

---

## 17. LSTM 的训练

与 `RNN` 的训练方式一致：

$$
\boxed{\text{BPTT}}
$$

即：`Backpropagation Through Time`，随时间反向传播。

---

## 18. LSTM 的核心总结

`LSTM` 的核心结构：
$$
f_t=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+b_f)
$$

$$
i_t=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_i)
$$

$$
\tilde{C}_t=\tanh(W_{xc}x_t+W_{hc}h_{t-1}+b_c)
$$

$$
C_t=f_t\odot C_{t-1}+i_t\odot\tilde{C}_t
$$

$$
o_t=\sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_o)
$$

$$
h_t=o_t\odot\tanh(C_t)
$$

一句话总结：**LSTM 通过细胞状态 $C_t$ 保存和传递长期信息，并利用遗忘门、输入门和输出门分别控制旧信息的保留、新信息的写入以及当前信息的输出，从而增强对长序列依赖关系的建模能力。**

