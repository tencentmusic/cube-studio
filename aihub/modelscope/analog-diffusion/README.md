
**Analog Diffusion**

![Header](images/page1.jpg)

本模型利用dreambooth方法微调Stable Diffusion 1.5而来，数据集为胶片摄影数据集。

在prompt中加入`analog style` 以实现胶片摄影效果。

在`negative_prompt` 中最好加入` blur haze naked` . 微调数据集中并不存在NSFW样本但是模型有可能输出NSFW内容。

请注意，在`negative_prompt` 中加入`blur`和`haze`，可以使图像更清晰，但也有不太明显的模拟胶片效果。



样例图像的参数（prompt, sampler, seed, *etc.*）请参考[本文件](parameters_used_examples.txt)。

![Environments Example](images/page2.jpg)
![Characters Example](images/page3.jpg)



 [点击此处查看一些随机挑选的样本](https://imgur.com/a/7iOgTFv)



## 使用方法：

```python
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2

pipe = pipeline(task=Tasks.text_to_image_synthesis, 
                model='dienstag/Analog-Diffusion',
                model_revision='v1.0')

prompt = 'analog style blonde Princess Peach in the mushroom kingdom'
negative_prompt = 'blur haze'
output = pipe({'text': prompt, 'negative_prompt': negative_prompt})
cv2.imwrite('result.png', output['output_imgs'][0])
```
