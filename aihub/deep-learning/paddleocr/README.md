# 1. OCR 技术

先简单介绍下OCR技术，它包含两个基本的神经网络模型。

一个是文字检测模型，它的本质是目标检测，能将一张图片中文字部分用矩形框框出来。

另一个是文字识别模型，文字检测模型框出来的部分，送入文字识别模型，可以识别出对应的文本。

下面我们用百度开源的PaddleOCR框架，实践一下OCR技术。

执行以下命令，安装PaddleOCR

```bash
pip install "paddleocr>=2.0.1"
```
安装完成后，编码调用ocr函数，提取下图中的文字

```bash
from paddleocr import PaddleOCR, draw_ocr

ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # lang="ch"代表识别中文
img_path = './imgs/12.png' # 上图
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)
```
首次运行可以看到以下输出

程序会自动下载三个模型文件，分别是ch_PP-OCRv3_det_infer.tar、ch_PP-OCRv3_rec_infer.tar和ch_ppocr_mobile_v2.0_cls_infer.tar。

前两个是我们刚刚提到的文字检测模型和文字识别模型。

最后一个是文本方向分类模型，因为创建PaddleOCR时指定了use_angle_cls=True，因此会下载该模型，我们暂时不关注它。

这三个模型都是PaddleOCR在自己的数据集上预训练好了的，对于常规的图片都可以直接用，如果大家有比较特殊的数据集需要识别，如：车牌，只能按照PaddleOCR提供的方法，自己训练模型。

程序最后一行print(line)输出的内容如下：

可以看到，成功能够提取出图片中的文本了，并且准确度没问题。

输出的每一行包含两个元素，第1个元素是文字检测模型识别出文本的矩形框坐标，第2个元素是文字识别模型，识别出来的文本内容以及对应的分数，分数越接近1说明越准确。

如果你觉得print打印出来的结果不直观，我们可以用可视化来显示结果

```bash
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

result.jpg如下：

左侧的红框是文字检测模型检测出来的文字，右侧是文字识别模型识别出来的文字以及对应的分数。
