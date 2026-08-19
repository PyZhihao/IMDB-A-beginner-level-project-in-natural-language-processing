# GRU（门控循环单元）

## 1. GRU 是什么

`GRU`，全称： $\text{Gated Recurrent Unit}$，中文叫做：门控循环单元

`GRU` 是一种用于处理**序列数据**的循环神经网络结构，改进传统 `RNN`。

与 `RNN` 类似，`GRU` 也通过隐藏状态： $h_t$ 将之前的信息传递到当前时间步。

但 `GRU` 在隐藏状态更新过程中引入了**门控机制**：

- 更新门（`Update Gate`）
- 重置门（`Reset Gate`）

---

## 2. 为什么需要 GRU

传统 `RNN` 为： $h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)$

每一个时间步都会直接根据： $x_t$ 和 $h_{t-1}$ 重新计算隐藏状态。

也就是说，**传统 `RNN` 没有专门的机制决定哪些历史信息应该保留，哪些应该遗忘**。在长序列中，随着信息不断传递，早期的重要信息可能逐渐丢失。

`GRU` 的核心思想就是：**在 `RNN` 的基础上加入门控机制，让模型自己学习应该保留多少历史信息，以及应该加入多少当前的新信息。**

---

## 3. GRU 的核心结构

`GRU` 主要包含：

1. 更新门（`Update Gate`）
2. 重置门（`Reset Gate`）
3. 候选隐藏状态（`Candidate Hidden State`）
4. 当前隐藏状态（`Hidden State`）

核心结构：

1. 更新门： $z_t=\sigma(W_{xz}x_t+W_{hz}h_{t-1}+b_z)$
2. 重置门： $r_t=\sigma(W_{xr}x_t+W_{hr}h_{t-1}+b_r)$
3. 候选隐藏状态： $\tilde h_t=\tanh\left(W_{xh}x_t+W_{hh}(r_t\odot h_{t-1})+b_h\right)$
4. 当前隐藏状态： $h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t$

------

## 4. 更新门（Update Gate）

更新门： $\boxed{z_t=\sigma(W_{xz}x_t+W_{hz}h_{t-1}+b_z)}$

其中：

- $x_t$：当前时间步输入
- $h_{t-1}$：上一时间步隐藏状态
- $W_{xz}$、 $W_{hz}$：可学习权重参数
- $b_z$：可学习偏置参数
- $\sigma$：`Sigmoid` 激活函数

`Sigmoid` 的输出范围为： $0<z_t<1$，因此更新门可以看成一个比例系数。

更新门的主要作用：**控制保留多少上一时间步隐藏状态 $h_{t-1}$ 和当前时间步候选隐藏状态 $\tilde h_t$ 信息进入当前时间步隐藏状态 $h_{t}$**

------

## 5. 重置门（Reset Gate）

重置门： $\boxed{r_t=\sigma(W_{xr}x_t+W_{hr}h_{t-1}+b_r)}$

其中：

- $x_t$：当前时间步输入
- $h_{t-1}$：上一时间步隐藏状态
- $W_{xr}$、 $W_{hr}$：可学习权重参数
- $b_r$：可学习偏置参数
- $\sigma$：`Sigmoid` 激活函数

同样： $0<r_t<1$

重置门的主要作用：**控制保留多少上一时间步隐藏状态 $h_{t-1}$ 进入当前时间步候选隐藏状态 $\tilde h_t$**

------

## 6. 候选隐藏状态

得到重置门之后，计算候选隐藏状态： $\boxed{\tilde h_t = \tanh\left(W_{xh}x_t+W_{hh}(r_t\odot h_{t-1})+b_h\right)}$

这里最重要的是： $r_t\odot h_{t-1}$，也就是说：**GRU 不会直接使用全部历史信息计算新的候选状态，而是先通过重置门进行筛选。**

因此： $\tilde h_t$ 可以理解为：**根据当前输入和筛选后的历史信息得到的“新的候选信息”。**

------

## 7. 当前隐藏状态

得到候选隐藏状态之后，需要计算当前的隐藏状态： $\boxed{h_t=z_t\odot h_{t-1}+(1-z_t)\odot \tilde h_t}$

**保留旧信息**： $z_t\odot h_{t-1}$

**加入新信息**： $(1-z_t)\odot\tilde h_t$

------

## 8. 如何理解更新门和重置门

重置门： $r_t$，用于计算候选隐藏状态： $\tilde h_t = \tanh\left(W_{xh}x_t+W_{hh}(r_t\odot h_{t-1})+b_h\right)$。

**重置门决定的是当前时间步候选隐藏状态 $\tilde h_t$ 中保留上一时间步隐藏状态 $h_{t-1}$ 的多少**

当 $r_t\approx0$，此时： $r_t\odot h_{t-1}\approx0$，说明主要保留当前信息，基本忽略旧隐藏状态的信息。

当 $r_t\approx1$，此时： $r_t\odot h_{t-1}\approx h_{t-1}$，说明主要保留旧隐藏状态的信息，基本忽略当前信息。



更新门： $z_t$，用于计算当前隐藏状态： $h_t=z_t\odot h_{t-1}+(1-z_t)\odot \tilde h_t$。

**更新门决定的是当前时间步隐藏状态中 $h_{t}$ 保留上一时间步隐藏状态 $h_{t-1}$ 和当前时间步候选隐藏状态 $\tilde h_t$ 的比例**

当 $z_t\approx0$，此时： $1-z_t\approx1$，即 $h_t\approx\tilde h_t$，说明主要使用当前新候选隐藏状态的信息，较少保留旧隐藏状态的信息。

当 $z_t\approx1$，此时： $1-z_t\approx0$，即 $h_t\approx h_{t-1}$，说明主要保留旧隐藏状态的信息，很少加入当前新候选隐藏状态的信息。

------

## 9. GRU 一个时间步的完整计算

给定：当前时间步输入 $x_t$ 和上一时间步隐藏状态 $h_{t-1}$

第一步：计算更新门： $z_t=\sigma(W_{xz}x_t+W_{hz}h_{t-1}+b_z)$

第二步：计算重置门： $r_t=\sigma(W_{xr}x_t+W_{hr}h_{t-1}+b_r)$

第三步：计算候选隐藏状态： $\tilde h_t=\tanh\left(W_{xh}x_t+W_{hh}(r_t\odot h_{t-1})+b_h\right)$

第四步：计算当前隐藏状态： $h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t$

------

## 10. GRU 如何处理整个序列

假设输入序列： $x_1,x_2,x_3,\cdots,x_T$

首先初始化 $h_0$，一般初始化为全 $0$

然后随时间步进行隐藏状态的更新：
$$
(x_1,h_0)\rightarrow h_1
$$

$$
(x_2,h_1)\rightarrow h_2
$$

$$
(x_3,h_2)\rightarrow h_3
$$

$$
\cdots
$$

$$
(x_T,h_{T-1})\rightarrow h_T
$$
```text
x1 → GRU → h1
            ↓
x2 → GRU → h2
            ↓
x3 → GRU → h3
            ↓
           ...
            ↓
xT → GRU → hT
```

每一个时间步内部都会计算：更新门 $z_t$；重置门 $r_t$；候选隐藏状态 $\tilde h_t$；当前隐藏状态 $h_t$。

------

## 11. GRU 的参数共享

**GRU 在所有时间步共享同一套参数。**

例如：
$$
W_{xz},W_{hz}
$$

$$
W_{xr},W_{hr}
$$

$$
W_{xh},W_{hh}
$$

------

## 12. PyTorch 中的 GRU

```python
self.gru = nn.GRU(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    num_layers=1,
    bias=True,
    batch_first=True,
    dropout=0.0,
    bidireactional=False
)
```

主要参数：

| 参数            | 含义                      |
| --------------- | ------------------------- |
| `input_size`    | 每个时间步输入特征的维度  |
| `hidden_size`   | 隐藏状态的维度            |
| `num_layers`    | GRU 的层数                |
| `bias`          | 是否使用偏置              |
| `batch_first`   | 是否让 Batch 位于第一维   |
| `dropout`       | 多层 GRU 层之间的 Dropout |
| `bidirectional` | 是否使用双向结构          |

------

## 13. GRU 的输入 Shape

`GRU` 的输入和 `RNN` 的输入一致。

##### 当设置 `batch_first=True` 

输入 `x` 的 `Shape`：

$$
\boxed{[\mathrm{batch\_size},\mathrm{seq\_len},\mathrm{input\_size}]}
$$

例如：

```text
batch_size = 32
seq_len = 200
embedding_dim = 128
```

`x.shape`为：`[32, 200, 128]`

含义：

- `32`：一个 `Batch` 中有 `32` 个样本
- `200`：每个序列有 `200` 个时间步（时间步大小）
- `128`：每个时间步输入向量维度为 `128`

------

##### 当设置`batch_first=False`

输入`x` 的 `Shape` 为：

$$
\boxed{[\mathrm{seq\_len},\mathrm{batch\_size},\mathrm{input\_size}]}
$$

例如：

```python
batch_size = 32
seq_len = 200
embedding_dim = 128
```

`x.shape` 为：`[200, 32, 128]`

含义分别是：

- `200`：每个序列长度为 `200`，即时间步大小
- `32`：一个 `Batch` 中有 `32` 个样本（时间步大小）
- `128`：每个时间步输入向量维度为 `128`

`batch_first` **只会交换前两个维度的位置**，不会改变数据本身的含义。

另外要注意：`batch_first` 只影响输入 `x` 和输出 `output` 的维度顺序，不影响 `h_n` 的维度顺序。

------

## 14. GRU 的输出

调用：`output, h_n = self.gru(x)`

会返回两个结果：

```python
output, h_n = self.gru(x)
```

返回：

```text
output
h_n
```

---

### 14.1 output

当`batch_first=True`，且为单向 `GRU` 时，`output`的`shape`为：

$$
\boxed{[\mathrm{batch\_size},\mathrm{seq\_len},\mathrm{hidden\_size}]}
$$

例如：

```text
[32, 200, 128]
```

`output` 表示：**最后一层 GRU 在所有时间步产生的隐藏状态。**也就是： $[h_1,h_2,\cdots,h_T]$

例如：`output[:, 0, :]` 表示最后一层第一个时间步的隐藏状态： $h_1$

`output[:, -1, :]`，表示最后一层最后一个时间步的隐藏状态： $h_T$

---

### 14.2 `h_n`

`h_n` 的`shape` 为：

$$
\boxed{[\mathrm{num\_layers},\mathrm{batch\_size},\mathrm{hidden\_size}]}
$$

例如：`[2, 32, 128]`

`h_n` 表示：**每一层 GRU 在最后一个时间步的隐藏状态。**

例如：`h_n[-1]`表示：最后一层 `GRU` 在最后一个时间步的隐藏状态，`shape`为：

$$
\boxed{[\mathrm{batch\_size},\mathrm{hidden\_size}]}
$$

---

## 15. 多层 GRU

如果：`num_layers=2`，那么就是两层 `GRU`。

第一层： $x_t \rightarrow h_t^{(1)}$

第二层接收第一层的输出： $h_t^{(1)}\rightarrow h_t^{(2)}$

每一层还会维护自己的历史隐藏状态： $h_t^{(1)}=f(x_t,h_{t-1}^{(1)})$； $h_t^{(2)}=f(h_t^{(1)},h_{t-1}^{(2)})$

```text
             时间方向 →

Layer 2:    h1² → h2² → h3² → h4²
             ↑     ↑     ↑     ↑
Layer 1:    h1¹ → h2¹ → h3¹ → h4¹
             ↑     ↑     ↑     ↑
            x1    x2    x3    x4
```

因此多层 `GRU` 中有两个方向的信息传递：

- 横向：时间步之间的信息传递
- 纵向：网络层之间的信息传递

------

## 16. GRU 的训练

与 `RNN` 的训练方式一致：

$$
\boxed{\text{BPTT}}
$$

即：`Backpropagation Through Time`，随时间反向传播。

***

## 17. GRU 的核心总结

`GRU`的核心结构：
$$
z_t=\sigma(W_{xz}x_t+W_{hz}h_{t-1}+b_z)
$$

$$
r_t=\sigma(W_{xr}x_t+W_{hr}h_{t-1}+b_r)
$$

$$
\tilde h_t=\tanh\left(W_{xh}x_t+W_{hh}(r_t\odot h_{t-1})+b_h\right)
$$

$$
h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t
$$

一句话总结：**GRU 通过合并门控机制（重置门与更新门）来简化网络结构，在有效缓解梯度消失问题的同时，显著提升了计算效率。**
