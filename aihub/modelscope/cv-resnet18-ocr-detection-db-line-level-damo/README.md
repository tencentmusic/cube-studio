

# DBNet文字检测行检测模型介绍
文字检测，即给定一张图片，检测出图中所含文字的外接框的端点的坐标值。文字行检测即检测给定图片中文字行的外接框。

本模型用于通用场景的行检测，我们还有下列用于其他场景的模型：
- [通用场景行检测](https://modelscope.cn/models/damo/cv_resnet18_ocr-detection-line-level_damo/summary)
- [通用场景单词检测](https://modelscope.cn/models/damo/cv_resnet18_ocr-detection-word-level_damo/summary)

文本识别模型：
- [通用场景](https://modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-general_damo/summary)
- [手写场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/summary)
- [文档印刷场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-document_damo/summary)
- [自然场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-scene_damo/summary)
- [车牌场景](https://www.modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-licenseplate_damo/summary)

以及对整图中文字进行检测识别的完整OCR能力：
- [通用场景整图检测识别](https://modelscope.cn/studios/damo/cv_ocr-text-spotting/summary)

## 模型描述

本模型是基于分割的文字检测方法，把文字行的区域分割文字中心区域和文字边界区域，通过处理得到文字完整区域，最后得到文字区域的外接框。详见：[DBNet(Paper)](https://arxiv.org/pdf/1911.08947.pdf)


## 期望模型使用方式以及适用范围
本模型主要用于给输入图片输出图中文字外接框坐标，具体地，模型输出的框的坐标为文字框四边形的四个角点的坐标，左上角为第一个点，按照顺时针的顺序依次输出各个点的坐标，分别为(x1,y1)(x2,y2)(x3,y3)(x4,y4)。用户可以自行尝试各种输入图片。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope之后即可使用ocr-detection的能力。

### 预处理和后处理
测试时的主要预处理和后处理如下：
- Resize Pad（预处理）: 输入图片短边resize到1736，短边等比例缩放，并且补pad到长短边相等
- threshold后处理）: thresh和box_thresh采用0.2和0.3的threshold

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-db-line-level_damo')
result = ocr_detection('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg')
print(result)
```

### 完整OCR能力体验
如果想体验完整的OCR能力，对整图中的文字进行检测识别，可以体验[创空间应用](https://modelscope.cn/studios/damo/cv_ocr-text-spotting/summary)。对于文字检测模型和文字识别模型的串联，可以参考[说明文档](https://modelscope.cn/dynamic/article/42)。

### 模型局限性以及可能的偏差
- 模型是在特定中英文数据集上训练的，在其他场景和语言的数据上有可能产生一定偏差，请用户自行评测后决定如何使用。
- 当前版本在python3.7环境CPU和单GPU环境测试通过，其他环境下可用性待测试

## 训练数据介绍
本模型行检测模型训练数据集是MTWI/ReCTS/SROIE/LSVT，训练数据数量约53K。

## 模型训练流程
本模型利用imagenet预训练参数进行初始化，然后在训练数据集上进行训练，先利用640x640尺度训练200epoch。

#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/damo/cv_resnet18_ocr-detection-db-line-level_damo.git
```