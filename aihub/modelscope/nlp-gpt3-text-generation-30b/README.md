
# GPT3中文30B参数量文本生成模型
GPT-3模型是一个通用的预训练生成模型，使用Transformer的Decoder-only结构，可以用于解决下游各种类型的生成任务，特别是zero-shot生成能力。模型利用大量无监督数据，通过自回归任务进行预训练。可以用于解决文本生成相关的任务包含：文本摘要、问题生成、data-to-text等。

**Demo体验，请点击右侧进入AI写手创空间!!!**

## 模型描述
GPT-3模型使用Transformer的 Decoder结构，并对Transformer Decoder进行了一些改动，原本的Decoder包含了两个 Multi-Head Attention 结构，GPT-3只保留了 Mask Multi-Head Attention，利用常规的语言建模优化，从左到右的自回归预训练。本模型是基于GPT-3的代码结合大量中文无监督数据和下游任务数据预训练得到，我们训练了多种不同参数的模型，此处展示的是GPT-3 300亿参数模型。GPT-3模型介绍，详见：[Language Models are Few-Shot Learners
](https://arxiv.org/abs/2005.14165)

本项目我们复现了一系列不同规模的中文GPT3模型，包括base/large/1.3B/2.7B/13B/30B/175B等，本模型是其中30B的版本。全部版本如下表所示：

|Model|Layers|Heads|d_model|LR|Batch|
|---|---|---|---|---|---|
|[base](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_chinese-base/summary)|12|12|768|6.0e-4|0.5M|
|[large](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_chinese-large/summary)|24|16|1024|3.0e-4|0.5M|
|[1.3B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_1.3B/summary)|24|32|2048|2.0e-4|2M|
|[2.7B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_2.7B/summary)|32|32|2560|1.6e-4|2M|
|[13B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_13B/summary)|40|40|5120|1.0e-4|6M|
|[30B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_30B/summary)|48|56|7168|1.0e-4|6M|
|175B(work in process)|96|96|12288|1.2e-4|6M|

## 期望模型使用方式以及适用范围
本模型可直接用于文本生成，也可以通过finetune用于各类文本理解的任务。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
因模型规模太大，暂不支持下载，可以先使用右侧demo进行测试体验

### 模型局限性以及可能的偏差
模型训练数据来源于网络，生成结果可能存在一定偏差。

## 训练数据介绍
训练数据包括中文维基百科、网络上公开文本数据。

## 模型训练流程
本模型的训练分为两个阶段。第一阶段严格按照原始GPT3的参数设置进行：在中文wiki/ Common crawl等无监督数据上，通过自回归的训练任务训练了约300B字得到。第二阶段中，我们加入了多种有监督数据继续训练，使得模型具备多种任务的zero-shot的处理能力。

### 预处理
暂无

### 训练
暂无

### 推理加速
我们对大规模生成模型的推理速度进行了极致优化，30B模型128字的文本生成可以在1.8秒左右完成。

### 数据评估及结果
您可以在本页面右侧的在线测试中体验本模型的生成效果。输入测试文本，点击执行测试。其中，有两个可配置的参数：
- max_length: 最大长度设置，包含了输入和输出加起来的总最大长度
- top_k: sampling decode生成的配置参数，用于确定当前生成从词表概率分布的top_k进行采样，该值设置的越大，模型生成的多样性越丰富，但同时相关性和准确性会有所下降

模型可以应用于作文生成，代码生成，小说续写，诗词生成等，以下展示了几个模型结果的示例：
- 作文生成
```
输入：以《我的家乡杭州》为题写一篇作文

输出：我的家乡杭州,有一个美丽的名字——天堂。 
杭州有许多美丽景点,比如雷锋夕照。雷锋夕照在哪?在市民广场,那里树木郁郁葱葱,鲜花五彩缤纷、争奇斗艳,是一个美丽的大花坛。每当夕阳西下,人们都会聚集在这里,拍照留念。还有美丽的断桥。在西湖的北面,有一座桥,叫断桥,断桥上有一座很长很长的断桥,人们都喜欢来这里拍照,感受那美丽的景色。 
杭州还有许多风味小吃。比如西湖醋鱼,外婆家的招牌小吃,是杭州的特色名菜。还有杭州的小笼包和牛肉面,味道都很美味。 杭州还有很多著名的特产,比如西湖龙井茶叶、西湖藕粉、西湖莼菜汤......都是杭州的特产,很好吃。 
杭州的风景美丽,物产丰富。我喜欢我的家乡,欢迎大家来杭州玩。

```

- 代码生成
```
输入：写一个python的快排代码

输出：
python3
class Solution:
 def quicksort(self, nums: List[int], left: int, right: int) -> List[int]:
  if not right:
   return nums
  i = left
  j = right
  k = left
  p = left
  while i < j:
   p = i
   while p > j and nums[p] < nums[j]:
    p = p - 1
   nums[p] = nums[j] 
   j -= 1
   nums[i] = nums[j]
   i += 1
   if p > k:
    nums[k] = nums[j]
   else:
    nums[k] = nums[p]
    p -= 1
   k += 1
  nums[k] = nums[i]
  return nums
```

- 生活百科
```
输入：西红柿炒胡萝卜的做法：

输出：1.胡萝卜洗净去皮切丝2.锅中加水烧开,加入胡萝卜丝焯水,水里加入适量食盐,焯过水的胡萝卜丝过凉水后备用3.西红柿去皮切成小块,蒜切末。4.锅中加入油,油热后放入蒜末和西红柿,翻炒出香味儿。5.放入胡萝卜丝翻炒6.放入焯过水的胡萝卜丝翻炒7.放入盐、鸡精,翻炒均匀即可关火8.出锅,撒上葱花,一盘香喷喷儿的西红柿炒胡萝卜就做好了。小诀窍:焯过水的胡萝卜丝一定要放入凉水中,焯过水的胡萝卜丝会更脆,口感更好9.西红柿炒胡萝卜可以加入少许的白糖,这样炒出的西红柿更加的甜,口感更佳!
```

- 诗词生成
```
输入：寒随穷律变，春逐鸟声开。初风飘带柳，

输出：寒随穷律变,春逐鸟声开。初风飘带柳,细雨湿花苔。
```

## 相关引用
我们将尽快推出本项目的技术报告，敬请期待！
