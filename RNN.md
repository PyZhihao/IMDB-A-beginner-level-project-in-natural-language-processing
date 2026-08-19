# RNN（循环神经网络）

## 1. RNN 是什么

`RNN`，全称： $\text{Recurrent Neural Network}$，中文叫做：循环神经网络

`RNN` 主要用于处理**序列数据**，例如：

- 文本序列
- 时间序列
- 语音序列

`RNN` 的特点是：在处理当前输入时，不仅考虑当前输入，还会考虑之前的信息。

---

## 2. RNN 的核心思想

`RNN` 使用隐藏状态 $h_t$ 来保存之前序列的信息。

核心关系： $\boxed{h_t=f(x_t,h_{t-1})}$

其中：

- $x_t$：当前时间步的输入
- $h_{t-1}$：上一时间步的隐藏状态
- $h_t$：当前时间步的隐藏状态

可以理解为：**当前状态 = 当前输入 + 历史信息**

---

## 3. RNN 的基本公式

$\boxed{h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)}$

其中：

- $W_{xh}$， $W_{hh}$：可学习权重参数
- $b_h$：可学习偏置参数
- $\tanh$：非线性激活函数

`tanh` 的输出范围为： $-1<h_t<1$，

---

## 4. RNN 如何处理序列

假设输入序列为： $x_1,x_2,x_3,\cdots,x_T$

`RNN`按照顺序进行计算。

首先： $h_1=f(x_1,h_0)$

然后： $h_2=f(x_2,h_1)$

接着： $h_3=f(x_3,h_2)$

一直到： $h_T=f(x_T,h_{T-1})$

其中： $h_0$为初始隐藏状态，一般初始化为全 $0$。

整体过程可以表示为：

```text
x1 → RNN → h1
            ↓
x2 → RNN → h2
            ↓
x3 → RNN → h3
            ↓
           ...
            ↓
xT → RNN → hT
```

------

## 5. 隐藏状态的含义

隐藏状态： $h_t$，可以理解为：`RNN` 处理到第 $t$ 个位置时，对之前序列信息的压缩表示。

$h_t=f(x_t,h_{t-1})$

而： $h_{t-1}=f(x_{t-1},h_{t-2})$

继续展开： $h_t=f(x_t,f(x_{t-1},f(x_{t-2},\cdots)))$

因此： $h_t$会受到： $x_1,x_2,\cdots,x_t$的影响。

所以可以记为：**截至当前位置的序列信息表示**

------

## 6. RNN 的参数共享

`RNN` 沿着时间展开之后，看起来好像有很多个 `RNN`：

```text
x1 → RNN → h1
x2 → RNN → h2
x3 → RNN → h3
```

但实际上：所有时间步使用的是同一个 `RNN` 单元。

也就是说，所有时间步共享同一套参数： $W_{xh}$，$W_{hh}$

这样做的好处是：

- 参数量不会随着序列长度增加
- 可以处理不同长度的序列
- 每个位置使用相同的序列处理规则

------

## 7. 文本经过 RNN 的过程

例如一个文本：

```text
I love this movie
```

首先转换为 `Token ID`：

```text
I      → 10
love   → 25
this   → 13
movie  → 67
```

然后经过 `Embedding`： $\text{Token ID}\rightarrow\text{Embedding Vector}$

得到： $x_1,x_2,x_3,x_4$

再依次输入 `RNN`：
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
(x_4,h_3)\rightarrow h_4
$$

完整流程：

```text
Text
 ↓
Token ID
 ↓
Embedding
 ↓
RNN
 ↓
Hidden State
```

------

## 8. PyTorch 中的 RNN

基本定义：

```python
self.rnn = nn.RNN(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    num_layers=1,
    nonlinearity = "tanh",
    bias = True,
    batch_first=True
    dropout = 0.0,
    bidirectional = False
)
```

主要参数：

| 参数            | 含义                              |
| --------------- | --------------------------------- |
| `input_size`    | 每个时间步输入特征的维度          |
| `hidden_size`   | 隐藏状态的维度                    |
| `num_layers`    | `RNN` 的层数                      |
| `nonlinearity`  | 每层`RNN`使用的非线性激活函数     |
| `bias`          | 是否使用偏置                      |
| `batch_first`   | 是否让 `batch` 维位于第一维       |
| `dropout`       | 多层 `RNN` 层与层之间的 `Dropout` |
| `bidirectional` | 是否使用双向 `RNN`                |

------

## 9. RNN 的输入 Shape

##### 当`batch_first=True`

输入`x` 的 `Shape` 为： $\boxed{[\text{batch_size},\text{seq_len},\text{input_size}]}$

例如：

```text
batch_size = 32
seq_len = 200
embedding_dim = 128
```

则`x.shape` 为：`[32, 200, 128]`

含义分别是：

- $32$：一个 `Batch` 中有 `32` 个样本
- $200$：每个序列长度为 `200`（时间步大小）
- $128$：每个时间步输入向量维度为 `128`

---

##### 当`batch_first=False`

输入`x` 的 `Shape` 为： $\boxed{[\text{seq_len},\text{batch_size},\text{input_size}]}$

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

## 10. RNN 的输出

调用：`output, h_n = self.rnn(x)`

会返回两个结果：

```text
output
h_n
```

------

### 10.1 `output`

对于单向 `RNN` ，`output`的`shape`为： $\boxed{[\text{batch_size},\text{seq_len},\text{hidden_size}]}$

例如：`[32, 200, 128]`

`output` 表示：**最后一层 `RNN` 在所有时间步产生的隐藏状态**，也就是： $[h_1,h_2,\cdots,h_T]$。

例如：`output[:, 0, :]` 表示最后一层第一个时间步的隐藏状态： $h_1$

`output[:, -1, :]`，表示最后一层最后一个时间步的隐藏状态： $h_T$

------

### 10.2 `h_n`

`h_n` 的`shape` 为： $\boxed{[\text{num_layers},\text{batch_size},\text{hidden_size}]}$

例如：`[2, 32, 128]`

`h_n` 表示：**每一层 `RNN` 在最后一个时间步的隐藏状态**。

例如：`h_n[-1]`表示：最后一层 `RNN` 的最后一个时间步的隐藏状态，`shape`为： $\boxed{[\text{batch_size},\text{hidden_size}]}$

------

## 11. 多层 RNN

如果：`num_layers=2`，那么就是两层 `RNN`。

第一层： $x_t\rightarrowh_t^{(1)}$

第二层接收第一层的输出： $h_t^{(1)}\rightarrowh_t^{(2)}$

每一层还会维护自己的历史隐藏状态： $h_t^{(1)}=f(x_t,h_{t-1}^{(1)})$；$ h_t^{(2)}=f(h_t^{(1)},h_{t-1}^{(2)})$

```text
             时间方向 →

Layer 2:    h1² → h2² → h3² → h4²
             ↑     ↑     ↑     ↑
Layer 1:    h1¹ → h2¹ → h3¹ → h4¹
             ↑     ↑     ↑     ↑
            x1    x2    x3    x4
```

因此多层 `RNN` 中有两个方向的信息传递：

- 横向：时间步之间的信息传递
- 纵向：网络层之间的信息传递

------

## 12. RNN 的训练

`RNN` 训练时仍然采用：`前向传播 + 反向传播`

由于 `RNN` 在时间维度上展开，所以反向传播也会沿着时间进行。

例如：

```text
x1 → h1 → h2 → h3 → h4
```

反向传播：

```text
Loss
 ↓
h4
 ↓
h3
 ↓
h2
 ↓
h1
```

这种方式称为：`Backpropagation Through Time`，简称：$\boxed{\text{BPTT}}$

------

## 14. RNN 的梯度问题

传统`RNN` 在序列较长时，反向传播会经过很多时间步。梯度中会出现连续相乘： $\frac{\partial h_t}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial h_{t-2}}\cdots\frac{\partial h_2}{\partial h_1}$，可能会出现两种问题：梯度消失和梯度爆炸

**梯度消失**：如果连续相乘的数值小于 $1$，例如： $0.5^{100}\approx0$，梯度会越来越小，从而产生梯度消失。进而导致网络浅层无法学习，训练停滞。解决方案：创新模型结构，例如`GRU`和`LSTM`。

**梯度爆炸**：如果连续相乘的数值大于 $1$，例如： $1.5^{100}$，梯度会快速增大，从而产生梯度爆炸。进而导致数值溢出，模型训练崩溃。解决方案：添加训练策略，例如训练过程中采用`梯度剪裁`。

------

## 15. RNN 核心总结

`RNN` 最核心的公式：
$$
h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
$$
一句话总结：**RNN通过引入隐藏状态，将序列历史信息随时间步传递，进而有效处理序列数据并捕获上下文依赖关系。**

