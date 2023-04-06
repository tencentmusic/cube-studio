
# ICASSP2023 MUG Challenge Track1 文本话题分割Baseline

## 赛事及背景介绍
随着数字化经济的进一步发展，越来越多的企业开始将现代信息网络作为数据资源的主要载体，并通过网络通信技术进行数据传输。同时，疫情也促使越来越多行业逐步将互联网作为主要的信息交流和分享的方式。
以往的研究表明，会议记录的口语语言处理（SLP）技术如关键词提取和摘要，对于信息的提取、组织和排序至关重要，可以显著提高用户对重要信息的掌握效率。
本项目源自于ICASSP2023信号处理大挑战的通用会议理解及生成挑战赛（MUG challenge），赛事构建并发布了目前为止规模最大的中文会议数据集，并基于会议人工转写结果进行了多项SLP任务的标注；
目标是推动SLP在会议文本处理场景的研究并应对其中的多项关键挑战，包括 人人交互场景下多样化的口语现象、会议场景下的长篇章文档建模等。

## 模型介绍
针对MUG挑战赛的赛道-话题分割任务，我们使用阿里巴巴达摩院自研模型[PoNet](https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary)构建了对应基线。

PoNet是一种具有线性复杂度(O(N))的序列建模模型，使用pooling网络替代Transformer模型中的self-attention来进行上下文的建模。
PoNet模型主要由三个不同粒度的pooling网络组成，一个是全局的pooling模块(GA)，分段的segment max-pooling模块(SMP)，和局部的max-pooling模块(LMP)，对应捕捉不同粒度的序列信息：
1.	在第一阶段，GA沿着序列长度进行平均得到句子的全局表征g。为了加强对全局信息的捕捉，GA在第二阶段对g和输入训练计算cross-attention。由于g的长度为1，因此总的计算复杂度仍为O(N)。
2.	SMP按每个分段求取最大值，以捕获中等颗粒度的信息。
3.	LMP沿着序列长度的方向计算滑动窗口max-pooling。
4.	然后通过池化融合(PF)将这些池化特征聚合起来。由于GA的特征在整个token序列是共享的，SMP的特征在segment内部也是共享的，直接将这些特征加到原始token上会使得token趋同（向量加法），而这种token表征同质化的影响将会降低诸如句子对分类任务的性能。
因此，我们在PF层将原始的token于对应的GA，SMP特征计算元素乘法得到新的特征，使得不同的token对应了不同的特征。

针对话题分割任务，segment是采用的段落粒度，对带段落的长文本进行中文话题分割。

赛道报名页面：https://modelscope.cn/competition/12/summary

## 使用方式
直接输入长篇的未分割文章，得到输出结果

## 模型局限性以及可能的偏差
模型采用AliMeeting4MUG Corpus语料进行训练，在其他领域文本上的话题分割性能可能会有偏差。

## 训练方式
模型用[nlp_ponet_fill-mask_chinese-base](https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary)初始化，在AliMeeting4MUG Corpus的话题分割训练数据上进行训练。初始学习率为5e-5，batch_size为2，max_seq_length=4096

More Details: https://github.com/alibaba-damo-academy/SpokenNLP

## 模型效果评估
在MUG的 Topic Segmentation的开发集结果如下：

| Model      | Backbone                                                                                                            | Positive F1 | 
|------------|---------------------------------------------------------------------------------------------------------------------|-------------|
| Longformer | IDEA-CCNL/Erlangshen-Longformer-110M                                                                                | 0.2324      |
| PoNet      | [damo/nlp_ponet_fill-mask_chinese-base](https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary) | 0.2517      |

注：Longformer是五个不同种子的平均结果。

## 代码范例
```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(
    task=Tasks.document_segmentation,
    model='damo/nlp_ponet_document-segmentation_topic-level_chinese-base')

paragraphs = ['哎大家好啊欢迎大家准时来参加我们的会议，今天我们会议的主题呢是如何提升我们这个洗发水儿的品牌影响力啊。我们现在是主要的产品是这个霸王防脱洗发水儿，现在请大家发表自己的意见啊欢迎大家努力的发表，请先从这位男士开始。是这个。', '我觉得，我觉得抖音应该挺火的，因为现在对，因为现在大家就是什么看电视什么的，都是看电视看报纸看杂志，从相对来说少一些。', '是吧。', '对可以找几个大V来那个推广咱们这洗发水儿，比如说发一些视频呀什么的是吧？洗发水的视频。', '但，抖音。', '对。', '但你。', '抖音的对现在是人最多的，也觉是整体。', '但抖音上，有一些那个广告儿就直接就是那种一刷刷到，嗯，对对对，就那种大家都比较反感。', '插入了是吧。', '对，那种广告便宜点儿。', '对。', '那大家看见咱们这种同品牌的类似的，这种洗发水儿的广告嘛。', '嗯。', '你说的啊，你说的是那种找那个大V在抖音上，然后直接推那种是吗？', '海飞。', '对推咱们的那个。', '不是，不是特早，我记得那个霸王就是成龙代言的，我就一下儿就把它给炒火了。然后那儿从那儿。', '对他是主要是明星的影响力比较大。', '我我觉。', '对明星影响力。', '对我觉得还是。', '而且好像他这个效果确实是也是挺不错的，而且我觉得挺好的。', '对我我就是用，我就用用霸王，你，其实我这都，嗯。', '淘宝。我。', '哎是不上过，之前的话，是不是在中央央视发播过广告了？', '就是好好多呢成龙，对，啊，他这找明星代言，就是紧跟影响力就上来了。', '嗯，对最早都是央视广告多。', '对对对，好多砸了好多钱了已经啊。', '我觉得。', '也挺好的，但是现在可能是不是看电视的人少，对淘宝也可以。', '烧钱啊。', '淘宝直播带货还是比较好一点儿，我觉得抖音这个这个平台就是太太那什么有点是有点low，对对对。', '嗯。是吧。', '淘淘宝现在，对。', '一二楼就是好多假的东西太多是吧，嗯。', '而且你你想防防脱的话好多就是上了年纪的人，但是他也不一定是。', '防脱。现在年轻人也都脱发了也挺多的，对你就是怎么能让咱。', '年轻人多点，脱发的很多，你看这个年纪，现在年轻人脱发年纪越来越越早，啊。', '啊。', '嗯。', '对吧？', '怎么能让咱们这个洗发水啊让更多的人知道，然后更多人接受会会有去尝试，我觉得应该。', '我我觉那种淘宝直播，找找那种大V带货直播这种。', '现在淘宝的影响力已经没有没有抖音，还有小红书小红书小红书。', '还。', '淘宝已经不行了，还还不如那个什，对呀，拼多多，哎小红，对京东都可以的。', '不行哈。', '嗯，这种，没有没有抖音，嗯。', '我觉得小红书也可以啊，那个小红书的群体，受众群体还是比较那个什么的。', '对，还有你刚才说的拼多多也是一个拼多多也可以。', '小红书比较高端。', '趁拼多多现在最近很火，啊。', '嗯。', '拼多多也是，我一般。', '拼多多大家用的也挺多的。', '唉，双十一了，马上双十一了，对双十一的话，这些平台都可以投。', '对，对啊，马上双十一。', '对对对。对搞点活动啊什么的。', '等还有呃，对。', '对啊，对送买点什么赠赠品啊，是不是？对啊，送的。', '对。', '买赠，啊对，对还有什么付定金吗现在不都是付尾款对主要是。', '促销满减。', '对。', '买一赠一啊，是啊就是，还可以那个半夜抢购呀，是不是？嗯。', '哦哦哦。', '嗯，嗯。', '对还得多少卡点儿。', '哦，我还有觉得就是那个马路边儿那公交公交站台，那的也是提升品牌影响力的，一个挺挺好的一个，因为现在其它媒体很少接触，就这个公交站台可能说那么。', '啊，那广告。', '哦那个也行，公交机场对对对。', '嗯。对还要哎。还要电梯，我觉得楼宇电梯也电梯广告。', '写写字楼的那个那个液晶屏，对，其实它使这个的话还是好多就是白领啊，或者二三十岁的就是男性脱着脱着脱发的非常多，也舍得花钱。', '高。那个地铁站地铁站离那那个投放那种广告啊。', '对。', '对。对。', '对，就比较大众化，比较大众对对对对对。', '唉你唉。', '对对别局限于哪一个群体哈，应该各个群体都有，年轻的年老的都得用这样才能咱们产品跟大家都知道，老头。', '对那个国贸地铁口儿，那个弄弄点儿大众的牌子，都有。', '对对对。对。老头老太太，我觉得就几乎就没有什么影响力了，他说就无所谓了，嗯。', '电，啊。', '对呀，对，但。', '那个电视台，唉电电视台广告你。', '老头老太太就看电视咱年轻人也不看电视都看抖音，什么小红书啊，什么拼多多。', '很少看电视，对，不多了。', '对对对。', '对呀。', '虽然不看，但是我觉得电视台就是有的时候，大家还是比如说那个优酷直播那一类的那种，啊，不是，就就是那个。', '嗯。', '啊对直播的还是挺那个综艺综艺也行现在综艺也挺火的，我觉得综艺也可以是一个方法对。', '对直播的，网剧平台更好一些，嗯。', '综艺综艺节目里啊，植入一些啊啊，对。', '对。嗯。', '爱奇艺啊，是不是？腾讯视频都可以。', '但是那个花费太大了是吧？', '对，说到花费之后，大家觉得哪个平台花费可能更高一点儿呢？就相对来说。', '这就是抖抖音。', '我我觉得电视台的花费会高一点吧，综艺节目那种在在没有没有。', '电视台最贵吗？我怎么觉得现在抖音那像什么李佳琪的那种也挺高啊，好几十万。', '电视台最高的。', '对，嗯，啊。嗯。', '抖音还有那个百度的那个老，那个浏览。', '哎，李佳琪在抖音上也有，也也也也，也在抖音上有做是吗？', '对。', '李佳琪也在抖音上也做啊在淘宝。', '对，前一阵儿那个江苏台有一个那什么为就那个晚会，都是他们那些大咖去的，然后中途也卖什么，什么产那个奥迪Q三就是插在那里头了，对，对，赞助那种平台。', '啊。', '那种。', '嗯。', '对但是赞助的那种就。', '对，带不直播，现在李湘不也是老是带货嘛，对吧？', '嗯，这是各个明星都靠带货了现在都不靠自己演出了。', '都在带货啊。', '啊，对对对对。', '对，都在带货，陈陈小春儿都在带货。', '带货好像据说他们的收入比那个什么还高，所以他们就得努力啊。', '嗯，但是咱们咱们这个品牌。', '唉赞赞助的话是不是收，呃花费比较高呀。', '对。', '赞助的话，应该不比带货便宜。', '哦带带货你得看谁带货，就是李佳琪和薇娅肯定那很很贵了。', '至少得。那个高那头部的。', '对可以找一些新人是吧？找个新人来带货更好。', '新人的话就是就影响力就差一点啦。', '但影响力，可能就怕不够吧。', '他有一些像比如说像抖音呢，像快手啊，他们那些就是那个大V，他其实并不是名人。', '嗯。', '啊对对对有一些唉你们有没有关注过一些农村的一些那些那些大姐啊，那什么的哈。', '对就是普通人。', '对。', '他们有些人的话。', '嗯，请一些比较小众的，对。', '对农村题材是吧？', '你这，其实。', '啊对就是比较那沙雕的那种是吗。', '什么农村创业啊什么的。', '其实你们其实我看中好多一些吃播的一些大V只要一开始说吃，吃着吃着然后就把那个，他所代言别的品牌的那个什么毒啊什么的，他就不说吃的他把这，他只是一切切入点，比如说哎。', '对对对对对对。', '就。', '广告拉出来了。', '就不说吃的了是吗？哦就吸引人眼球的是吧。', '你这胡子什么的，今儿去哪儿吃饭是不是刮了他的，把吉列的刮胡刀儿然后就推出来了，对，对，对，他增，他是先增播他的流量，完了之后，流量上来之后他再插播一些创意啊。', '哦哦哦。', '对。', '啊。', '对其实吃播就是先先增加那个粉丝流量嘛，是吧？', '对对对，好多，嗯。', '对对对。', '现在好多流量特别多的人，其实都是很普通的一个人，嗯。', '吃播。现在那流量最高的是哪个方面？是吃的方面还是用的还是什么玩儿的那种怎么能跟。', '都有，都有，现。', '都有都有，哎，但问题是，就是我们现在是选择平台嘛，就是是抖音呃。', '反正老有，有的人你要不关注。', '是是对，这是关键是咱们是一个防脱发洗发水儿，怎么能跟那些联系起来不是那么生硬最好。', '嗯。', '对。', '他就找一些比较时尚的，他经常的话说一些什么化妆品啊美妆啊，我觉得这些更合适。', '啊就是美妆博主呗，我光说那他们代言的品牌太杂了有点儿。', '嗯，前一个前一个。', '但但是您想你要找。', '白领。', '你要找，比如说薇娅和和李佳琪就太假了，我对对对，所以。', '嗯。', '不要找那样的，找那样的太贵了，啊，太费钱。', '嗯，其实还有一些电视剧那植入那会儿，就前一阵儿有一个那个伊能静的老公演艺的那个，就是叫秦昊演了一个脱发的一个人，就那的，对对对，那个挺挺火的，也可以就是软性的一些植入。', '哦哦哦，我知道那个隐秘的真相那个是吧。', '啊。对，可以。', '哦哦哦。', '隐秘的角落。', '你不是想找两个秃顶的那个来那个代理代言这个。', '对，他可能去演戏中人物是要这角色。', '我我觉我觉我觉得我们就是要从。', '还不如找葛优呢葛优挺好。', '葛优没头发，我觉得生发对他没意义。', '可是没头发。', '我觉得我们这个品牌影响力，要广告要从几个方面入手，就是都多那个，比如说第一个是抖音吧，第二个是那个电视台吧，第三个是那个带直播带货吧，是，线下的货都要全方面去去去做嘛啊。', '啊。', '嗯。', '啊对。', '户外户外，户，我觉得户外也是一种。', '线下的，户外的。', '都。就是全面铺开对吧，对就是重点有重点，然后分散。', '户外是全面覆盖。', '但某一个方面又得选一个那个呃选一个重点。', '对对对。可以开一些什么美发沙龙啊，之类的，也挺多的，嗯。', '重点，哎。', '对对对。', '其实您，不是。', '当然，每一个你选择的那个代言人也要有针对性，比如说抖音你选择的代言人是哪。', '对就是选择代言人人大家觉得选一个好呢还是多选几个呢？', '对，选个代言人。不选，也就选选，都选你的钱花得太多了，其实。', '我觉得多选几个，多选几个。', '选几个吧。', '但是我觉得你来一个重点儿的呗，然后找找几个大V。', '花个，不要。一个重点，然后两个辅助。', '啊。', '其其实我说一下，肯定你们都不知道不知道，就是就是唯一的现在一个男士，就是清扬以前C罗代言的，他要。', 'C罗都得得新冠了你还找你还找他。']

result = p(documents=paragraphs)

print(result[OutputKeys.TEXT])
```

### 相关论文以及引用信息
如果我们的模型对您有帮助，请您引用我们的文章：
```BibTex
@inproceedings{DBLP:journals/corr/abs-2110-02442,
  author    = {Chao{-}Hong Tan and
               Qian Chen and
               Wen Wang and
               Qinglin Zhang and
               Siqi Zheng and
               Zhen{-}Hua Ling},
  title     = {{PoNet}: Pooling Network for Efficient Token Mixing in Long Sequences},
  booktitle = {10th International Conference on Learning Representations, {ICLR} 2022,
               Virtual Event, April 25-29, 2022},
  publisher = {OpenReview.net},
  year      = {2022},
  url       = {https://openreview.net/forum?id=9jInD9JjicF},
}
```

# Baseline of ICASSP2023 MUG Challenge Track1 Topic Segmentation 

## Competition and background introduction
Meetings are vital for seeking information, sharing knowledge, and improving productivity. Technological advancements as well as the pandemic rapidly increased amounts of meetings. Prior studies show that spoken language processing (SLP) technologies on meeting transcripts, such as keyphrase extraction and summarization, are crucial for distilling, organizing, and prioritizing information and significantly improve users' efficiency in grasping salient information.
This project originated from the ICASSP2023 Signal Processing Grand Challenge - General Meeting Understanding and Generation (MUG) challenge. The event built and released the largest Chinese conference data set so far, and conducted a number of labeling of SLP tasks based on manual meeting transcripts.
The goal is to promote the research of SLP in meeting transcripts and address several key challenges, including diverse spoken language phenomena in human interaction scenarios, long-form document modeling in meeting scenarios, etc.

## Model description
For the track-topic segmentation task of the MUG Challenge, we used the self-developed model of Alibaba DAMO Academy [PoNet] (https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary) to construct a baseline .

PoNet is a Pooling Network (PoNet) for token mixing in long sequences with linear complexity, which uses a pooling network to replace the self-attention in the Transformer model for context modeling.
The PoNet model is mainly composed of three pooling networks with different granularities, one is the global pooling module (GA), the segment max-pooling module (SMP), and the local max-pooling module (LMP), corresponding to capturing different granularities of sequence. 
1. In the first stage, GA averages along the sequence length to obtain the global representation g of the sentence. In order to strengthen the capture of global information, GA calculates cross-attention for g and input training in the second stage. Since the length of g is 1, the total computational complexity is still O(N).
2. SMP is maximized per segment to capture moderately granular information.
3. LMP calculates the sliding window max-pooling along the direction of the sequence length.
4. These pooled features are then aggregated by Pooling Fusion (PF). Since the features of GA are shared throughout the token sequence, the features of SMP are also shared within the segment. Directly adding these features to the original token will make the token converge (vector addition), and the homogeneity of token representation will degrade performance on tasks such as sentence pair classification.
Therefore, we multiplied the original token with the corresponding GA and SMP feature calculation elements at the PF layer to obtain new features, so that different tokens correspond to different features.

For topic segmentation tasks, the paragraph is denoted as segment granularity. 

Track registration page: https://modelscope.cn/competition/12/summary

## How to use
Input long undivided articles and get output results

## Model limitations and possible bias
The model is trained with the AliMeeting4MUG Corpus, and the topic segmentation performance on texts in other fields may be biased.

## Training method
The model is initialized with [nlp_ponet_fill-mask_chinese-base](https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary) and trained on the topic segmentation training data of AliMeeting4MUG Corpus. 
The initial learning rate is 5e-5, batch size is 2, max_seq_length is 4096

More Details: https://github.com/alibaba-damo-academy/SpokenNLP

## Evaluation of model
The results of the development set of Topic Segmentation in MUG are as follows:

| Model      | Backbone                                                                                                            | Positive F1 | 
|------------|---------------------------------------------------------------------------------------------------------------------|-------------|
| Longformer | IDEA-CCNL/Erlangshen-Longformer-110M                                                                                | 0.2324      |
| PoNet      | [damo/nlp_ponet_fill-mask_chinese-base](https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary) | 0.2517      |

Note: Longformer is the average result of five different seeds.

## Code example
```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(
    task=Tasks.document_segmentation,
    model='damo/nlp_ponet_document-segmentation_topic-level_chinese-base')

paragraphs = ['哎大家好啊欢迎大家准时来参加我们的会议，今天我们会议的主题呢是如何提升我们这个洗发水儿的品牌影响力啊。我们现在是主要的产品是这个霸王防脱洗发水儿，现在请大家发表自己的意见啊欢迎大家努力的发表，请先从这位男士开始。是这个。', '我觉得，我觉得抖音应该挺火的，因为现在对，因为现在大家就是什么看电视什么的，都是看电视看报纸看杂志，从相对来说少一些。', '是吧。', '对可以找几个大V来那个推广咱们这洗发水儿，比如说发一些视频呀什么的是吧？洗发水的视频。', '但，抖音。', '对。', '但你。', '抖音的对现在是人最多的，也觉是整体。', '但抖音上，有一些那个广告儿就直接就是那种一刷刷到，嗯，对对对，就那种大家都比较反感。', '插入了是吧。', '对，那种广告便宜点儿。', '对。', '那大家看见咱们这种同品牌的类似的，这种洗发水儿的广告嘛。', '嗯。', '你说的啊，你说的是那种找那个大V在抖音上，然后直接推那种是吗？', '海飞。', '对推咱们的那个。', '不是，不是特早，我记得那个霸王就是成龙代言的，我就一下儿就把它给炒火了。然后那儿从那儿。', '对他是主要是明星的影响力比较大。', '我我觉。', '对明星影响力。', '对我觉得还是。', '而且好像他这个效果确实是也是挺不错的，而且我觉得挺好的。', '对我我就是用，我就用用霸王，你，其实我这都，嗯。', '淘宝。我。', '哎是不上过，之前的话，是不是在中央央视发播过广告了？', '就是好好多呢成龙，对，啊，他这找明星代言，就是紧跟影响力就上来了。', '嗯，对最早都是央视广告多。', '对对对，好多砸了好多钱了已经啊。', '我觉得。', '也挺好的，但是现在可能是不是看电视的人少，对淘宝也可以。', '烧钱啊。', '淘宝直播带货还是比较好一点儿，我觉得抖音这个这个平台就是太太那什么有点是有点low，对对对。', '嗯。是吧。', '淘淘宝现在，对。', '一二楼就是好多假的东西太多是吧，嗯。', '而且你你想防防脱的话好多就是上了年纪的人，但是他也不一定是。', '防脱。现在年轻人也都脱发了也挺多的，对你就是怎么能让咱。', '年轻人多点，脱发的很多，你看这个年纪，现在年轻人脱发年纪越来越越早，啊。', '啊。', '嗯。', '对吧？', '怎么能让咱们这个洗发水啊让更多的人知道，然后更多人接受会会有去尝试，我觉得应该。', '我我觉那种淘宝直播，找找那种大V带货直播这种。', '现在淘宝的影响力已经没有没有抖音，还有小红书小红书小红书。', '还。', '淘宝已经不行了，还还不如那个什，对呀，拼多多，哎小红，对京东都可以的。', '不行哈。', '嗯，这种，没有没有抖音，嗯。', '我觉得小红书也可以啊，那个小红书的群体，受众群体还是比较那个什么的。', '对，还有你刚才说的拼多多也是一个拼多多也可以。', '小红书比较高端。', '趁拼多多现在最近很火，啊。', '嗯。', '拼多多也是，我一般。', '拼多多大家用的也挺多的。', '唉，双十一了，马上双十一了，对双十一的话，这些平台都可以投。', '对，对啊，马上双十一。', '对对对。对搞点活动啊什么的。', '等还有呃，对。', '对啊，对送买点什么赠赠品啊，是不是？对啊，送的。', '对。', '买赠，啊对，对还有什么付定金吗现在不都是付尾款对主要是。', '促销满减。', '对。', '买一赠一啊，是啊就是，还可以那个半夜抢购呀，是不是？嗯。', '哦哦哦。', '嗯，嗯。', '对还得多少卡点儿。', '哦，我还有觉得就是那个马路边儿那公交公交站台，那的也是提升品牌影响力的，一个挺挺好的一个，因为现在其它媒体很少接触，就这个公交站台可能说那么。', '啊，那广告。', '哦那个也行，公交机场对对对。', '嗯。对还要哎。还要电梯，我觉得楼宇电梯也电梯广告。', '写写字楼的那个那个液晶屏，对，其实它使这个的话还是好多就是白领啊，或者二三十岁的就是男性脱着脱着脱发的非常多，也舍得花钱。', '高。那个地铁站地铁站离那那个投放那种广告啊。', '对。', '对。对。', '对，就比较大众化，比较大众对对对对对。', '唉你唉。', '对对别局限于哪一个群体哈，应该各个群体都有，年轻的年老的都得用这样才能咱们产品跟大家都知道，老头。', '对那个国贸地铁口儿，那个弄弄点儿大众的牌子，都有。', '对对对。对。老头老太太，我觉得就几乎就没有什么影响力了，他说就无所谓了，嗯。', '电，啊。', '对呀，对，但。', '那个电视台，唉电电视台广告你。', '老头老太太就看电视咱年轻人也不看电视都看抖音，什么小红书啊，什么拼多多。', '很少看电视，对，不多了。', '对对对。', '对呀。', '虽然不看，但是我觉得电视台就是有的时候，大家还是比如说那个优酷直播那一类的那种，啊，不是，就就是那个。', '嗯。', '啊对直播的还是挺那个综艺综艺也行现在综艺也挺火的，我觉得综艺也可以是一个方法对。', '对直播的，网剧平台更好一些，嗯。', '综艺综艺节目里啊，植入一些啊啊，对。', '对。嗯。', '爱奇艺啊，是不是？腾讯视频都可以。', '但是那个花费太大了是吧？', '对，说到花费之后，大家觉得哪个平台花费可能更高一点儿呢？就相对来说。', '这就是抖抖音。', '我我觉得电视台的花费会高一点吧，综艺节目那种在在没有没有。', '电视台最贵吗？我怎么觉得现在抖音那像什么李佳琪的那种也挺高啊，好几十万。', '电视台最高的。', '对，嗯，啊。嗯。', '抖音还有那个百度的那个老，那个浏览。', '哎，李佳琪在抖音上也有，也也也也，也在抖音上有做是吗？', '对。', '李佳琪也在抖音上也做啊在淘宝。', '对，前一阵儿那个江苏台有一个那什么为就那个晚会，都是他们那些大咖去的，然后中途也卖什么，什么产那个奥迪Q三就是插在那里头了，对，对，赞助那种平台。', '啊。', '那种。', '嗯。', '对但是赞助的那种就。', '对，带不直播，现在李湘不也是老是带货嘛，对吧？', '嗯，这是各个明星都靠带货了现在都不靠自己演出了。', '都在带货啊。', '啊，对对对对。', '对，都在带货，陈陈小春儿都在带货。', '带货好像据说他们的收入比那个什么还高，所以他们就得努力啊。', '嗯，但是咱们咱们这个品牌。', '唉赞赞助的话是不是收，呃花费比较高呀。', '对。', '赞助的话，应该不比带货便宜。', '哦带带货你得看谁带货，就是李佳琪和薇娅肯定那很很贵了。', '至少得。那个高那头部的。', '对可以找一些新人是吧？找个新人来带货更好。', '新人的话就是就影响力就差一点啦。', '但影响力，可能就怕不够吧。', '他有一些像比如说像抖音呢，像快手啊，他们那些就是那个大V，他其实并不是名人。', '嗯。', '啊对对对有一些唉你们有没有关注过一些农村的一些那些那些大姐啊，那什么的哈。', '对就是普通人。', '对。', '他们有些人的话。', '嗯，请一些比较小众的，对。', '对农村题材是吧？', '你这，其实。', '啊对就是比较那沙雕的那种是吗。', '什么农村创业啊什么的。', '其实你们其实我看中好多一些吃播的一些大V只要一开始说吃，吃着吃着然后就把那个，他所代言别的品牌的那个什么毒啊什么的，他就不说吃的他把这，他只是一切切入点，比如说哎。', '对对对对对对。', '就。', '广告拉出来了。', '就不说吃的了是吗？哦就吸引人眼球的是吧。', '你这胡子什么的，今儿去哪儿吃饭是不是刮了他的，把吉列的刮胡刀儿然后就推出来了，对，对，对，他增，他是先增播他的流量，完了之后，流量上来之后他再插播一些创意啊。', '哦哦哦。', '对。', '啊。', '对其实吃播就是先先增加那个粉丝流量嘛，是吧？', '对对对，好多，嗯。', '对对对。', '现在好多流量特别多的人，其实都是很普通的一个人，嗯。', '吃播。现在那流量最高的是哪个方面？是吃的方面还是用的还是什么玩儿的那种怎么能跟。', '都有，都有，现。', '都有都有，哎，但问题是，就是我们现在是选择平台嘛，就是是抖音呃。', '反正老有，有的人你要不关注。', '是是对，这是关键是咱们是一个防脱发洗发水儿，怎么能跟那些联系起来不是那么生硬最好。', '嗯。', '对。', '他就找一些比较时尚的，他经常的话说一些什么化妆品啊美妆啊，我觉得这些更合适。', '啊就是美妆博主呗，我光说那他们代言的品牌太杂了有点儿。', '嗯，前一个前一个。', '但但是您想你要找。', '白领。', '你要找，比如说薇娅和和李佳琪就太假了，我对对对，所以。', '嗯。', '不要找那样的，找那样的太贵了，啊，太费钱。', '嗯，其实还有一些电视剧那植入那会儿，就前一阵儿有一个那个伊能静的老公演艺的那个，就是叫秦昊演了一个脱发的一个人，就那的，对对对，那个挺挺火的，也可以就是软性的一些植入。', '哦哦哦，我知道那个隐秘的真相那个是吧。', '啊。对，可以。', '哦哦哦。', '隐秘的角落。', '你不是想找两个秃顶的那个来那个代理代言这个。', '对，他可能去演戏中人物是要这角色。', '我我觉我觉我觉得我们就是要从。', '还不如找葛优呢葛优挺好。', '葛优没头发，我觉得生发对他没意义。', '可是没头发。', '我觉得我们这个品牌影响力，要广告要从几个方面入手，就是都多那个，比如说第一个是抖音吧，第二个是那个电视台吧，第三个是那个带直播带货吧，是，线下的货都要全方面去去去做嘛啊。', '啊。', '嗯。', '啊对。', '户外户外，户，我觉得户外也是一种。', '线下的，户外的。', '都。就是全面铺开对吧，对就是重点有重点，然后分散。', '户外是全面覆盖。', '但某一个方面又得选一个那个呃选一个重点。', '对对对。可以开一些什么美发沙龙啊，之类的，也挺多的，嗯。', '重点，哎。', '对对对。', '其实您，不是。', '当然，每一个你选择的那个代言人也要有针对性，比如说抖音你选择的代言人是哪。', '对就是选择代言人人大家觉得选一个好呢还是多选几个呢？', '对，选个代言人。不选，也就选选，都选你的钱花得太多了，其实。', '我觉得多选几个，多选几个。', '选几个吧。', '但是我觉得你来一个重点儿的呗，然后找找几个大V。', '花个，不要。一个重点，然后两个辅助。', '啊。', '其其实我说一下，肯定你们都不知道不知道，就是就是唯一的现在一个男士，就是清扬以前C罗代言的，他要。', 'C罗都得得新冠了你还找你还找他。']

result = p(documents=paragraphs)

print(result[OutputKeys.TEXT])
```

## Related work and citation information
If our model is helpful to you, please cite our paper:
```BibTex
@inproceedings{DBLP:journals/corr/abs-2110-02442,
  author    = {Chao{-}Hong Tan and
               Qian Chen and
               Wen Wang and
               Qinglin Zhang and
               Siqi Zheng and
               Zhen{-}Hua Ling},
  title     = {{PoNet}: Pooling Network for Efficient Token Mixing in Long Sequences},
  booktitle = {10th International Conference on Learning Representations, {ICLR} 2022,
               Virtual Event, April 25-29, 2022},
  publisher = {OpenReview.net},
  year      = {2022},
  url       = {https://openreview.net/forum?id=9jInD9JjicF},
}
```

