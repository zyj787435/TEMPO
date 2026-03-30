## 一、输入文本任务的处理与数学化建模

### 1 问题背景与目标

本文中，用户以**纯自然语言(暂且只支持English)文本**的形式提供任务输入，该输入不包含任何显式的结构或标签信息，具有高度非结构化的特点。然而，为了支持后续的模型编织（Model Fabric）与工具选型（Tool Selection），Agent 需要将该文本任务转换为一种可计算、可比较的数学表示。

我们的目标是构建一个从文本任务到向量表示的映射过程，使其既能够表达任务的语义意图，又能够刻画其功能需求。

### **2 输入形式定义**

设用户输入的任务为一段文本：

$$T = \text{\{}w_{1},w_{2},\ldots,w_{n}\text{\}}$$

其中，$w_{i}$ 表示文本中的第 i 个词语。

### **3 文本规范化（Text Normalization）**

进行数学建模之前，我们首先对输入文本进行规范化处理，以减少语言噪声对后续建模的影响。该步骤包括：

- 统一大小写  
- 去除非语义性符号  
- 保留原始语义结构  

规范化后的文本记为：$\widetilde{T}$

#### 4 任务语义向量化（Semantic Task Embedding）

为刻画任务的整体语义意图。定义一个预训练的文本编码函数：

$$f_{\text{sem}}:\widetilde{T} \rightarrow R^{d}$$

<!-- **注：**这个函数用于把我们输入的文本经过计算变为一个d维向量，由我们预训练或者查阅论文等方式得到，目前待定，向量的内容具体保存哪些参数也待定。 -->

得到任务的语义向量表示：

$$z_{\text{sem}} = f_{\text{sem}}\left( \widetilde{T} \right)$$

计算方法：
利用Sentence-BERT（或等价的embedding 模型），python已经实现了相关的库，示例如下：
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def f_sem(text: str):
    # 输出 shape: (d,)
    return model.encode(text, normalize_embeddings=True)
```
#### 5 任务功能需求解析（Functional Signal Extraction）

仅依赖语义向量不足以支持精细化的 Agent 决策。因此，我们进一步从文本中解析任务的功能需求信号。

定义任务的功能属性向量为：

$$a_{T} = \left\lbrack a_{1},a_{2},\ldots,a_{k} \right\rbrack$$

**注：** 其中每一维对应一种潜在能力需求，如推理、编程、数学计算或外部工具依赖等。这些都需要我们定义，然后需要我们从文本信息中求出。

$a_{T}$的一种定义如下：

$$a_{T} = \left\lbrack a_{reason},a_{code},a_{math},a_{tool},a_{domain} \right\rbrack$$

| 维度       | 含义         | 直观解释                 |
|------------|--------------|--------------------------|
| reasoning  | 推理强度     | 是否需要多步逻辑         |
| coding     | 编程需求     | 是否要写代码             |
| math       | 数学计算     | 是否有公式/优化          |
| tool       | 工具依赖     | 是否需要外部 API         |
| domain     | 领域性       | 是否依赖专业领域知识     |

<!-- 然后我们去寻找可以计算$a_{T}$的函数之类的。 -->
计算方法：
$$a_i=\sigma(\sum_{\omega \in \widetilde{T}}I(\omega \in \mathcal{K}_i))$$

其中：
- $\mathcal{K}_i$ 第i个功能的关键词集合,$\mathcal{K}_i$我们需要通过对大量英文单词的归纳给出，我这里暂时取了一个很简单的集合，这也是导致答案不准确的重要原因
- $I(\cdot)$ 为指示函数，当括号内条件成立时取1，否则取0
- $\sigma(\cdot)=\frac{1}{1+e^{-x}}$ 为 Sigmoid 函数，用于将计数映射到 (0,1) 范围内


#### **6 多视角任务表示融合（Multi-View Task Representation）**

在获得语义表示和功能需求表示后，我们将二者进行融合，构造最终的任务表示向量：

$$z_{T} = \text{concat}\left( z_{\text{sem}},\lambda \cdot a_{T} \right)$$

其中 $\lambda$ 为权重系数，用于平衡语义信息与功能信号的重要性。
$\lambda$ 的取值如下：
$$\lambda \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$$
这里我们简单取1就行

#### 7 小结

整体任务处理流程可以概括为：

$$T \rightarrow \widetilde{T} \rightarrow z_{\text{sem}} \oplus a_{T} \rightarrow z_{T}$$

该任务表示构成了本文 Agent 框架中所有决策模块的共同输入基础。

## 二、基于 API-Bank为测试集 的工具选择实验方案

#### 1 实验目标

本实验旨在验证所提出的**基于任务需求表征的工具选择方法**，在标准工具选择数据集 **API-Bank** 上是否能够有效地从候选 API 集合中选择与任务需求最匹配的工具。

具体而言，本实验关注以下问题：

- 给定自然语言形式（仅限英语）的任务描述，模型是否能够选出与数据集中标注的正确 API？
- 相比于基于文本相似度或语义嵌入的方法，基于"任务需求--工具能力"匹配的方式是否具有优势？

#### 2 数据集说明（API-Bank）

**2.1 数据集来源**

API-Bank 由阿里巴巴 DAMO Academy 提供，专门用于评估自然语言任务到 API 的选择与检索能力。

- GitHub 地址：<https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank>

**2.2 数据内容**

API-Bank 中的每个样本通常包含以下信息：

- **Task**：自然语言形式的用户任务描述
- **APIs**：候选 API 列表（包含 API 名称与文本描述）
- **Ground Truth API**：人工标注的正确 API（或 API 序列）

在本实验中，仅使用 **单步 API 选择任务**，不涉及参数生成或真实 API 调用。

**2.3 数据集使用方式**

**附：API-Bank里面的数据集主要以json格式保存，大致的形式是这样：**

```json
{
  "query": "Get the weather of Beijing tomorrow",
  "apis": [
    {
      "name": "get_weather",
      "description": "Retrieve weather information for a given city and date"
    },
    {
      "name": "search_flight",
      "description": "Search available flights between two cities"
    }
  ],
  "gold_api": "get_weather"
}
```
验证方法很简单，把这个json数据里面的query读入，看看我们实现的模型给出的gold_api是否一样，name和description是否相似。

#### 3 实验整体流程

实验整体流程如下所示：

text
自然语言任务描述
↓
任务需求表征（需求向量 $a_{T}$以及其他数据等等）
↓
API 工具能力表征（求能力向量$a_{API}$）
↓
需求--能力匹配（相似度计算）
↓
API 排序
↓
Top-k 预测结果


#### 4 API 工具能力表征（Tool Representation）
**4.1 输入**

API-Bank 中每个 API 的文本描述（功能说明、用途描述等）

**4.2 表征方式**

对 API 描述文本采用与任务需求解析相同的映射规则，构造 API 的能力向量：

$$a_{API} = \left\lbrack a_{reason},a_{code},a_{math},a_{tool},a_{domain} \right\rbrack$$

这里的计算方法和$a_{T}$是一样的

这样就便于我们接下来做相似度的计算了

#### 5 需求--能力匹配方法（Matching）
**5.1 相似度计算**

对于每一个任务--API 对，计算其需求--能力匹配得分：

$$score(T,API_i)=cosine\_similarity(a_{T},a_{API_i})$$


这里我们计算直接利用已有的scikit-learn库就行，示例如下：
```python
from sklearn.metrics.pairwise import cosine_similarity

score = cosine_similarity(
    a_T.reshape(1, -1),
    a_API.reshape(1, -1)
)[0][0]

```
**6.2 API 排序策略**

对候选 API 集合逐一计算匹配得分

按得分从高到低进行排序

排序结果作为工具选择的预测输出

#### 7 Baseline 方法

（下面就是对我们实现好的模型进行验证的一些方法，暂时没看懂多少，感觉也不一定用这些，等前面的实现完了能跑了再修改也不迟）

为了验证方法的有效性，在 API-Bank 上设置以下 baseline 方法进行对比：

**7.1 文本检索类方法**

BM25

TF-IDF

**7.2 语义匹配类方法**

Sentence-BERT 余弦相似度

基于 LLM embedding 的相似度匹配

**7.3 方法对比表**

方法	描述
BM25	关键词匹配
SBERT	语义相似度
LLM-Select	LLM 直接预测 API
Ours	任务需求--工具能力匹配
#### 8 评价指标（Evaluation Metrics）

采用 API-Bank 中常用的工具选择评估指标：

- Top-1 Accuracy

- Top-3 Accuracy

- Top-5 Accuracy

定义如下：

$$
\text{Accuracy@k} = \frac{1}{N} \sum_{i=1}^{N} I(\text{GroundTruth API}_i \in \text{Top-k predictions}_i)
$$

#### 9 实验实现细节（Implementation Details）

编程语言：Python

文本处理：NLTK / spaCy

向量计算：NumPy

相似度计算：scikit-learn