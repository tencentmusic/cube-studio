# 国画Diffusion
This is the fine-tuned Stable Diffusion model trained on traditional Chinese paintings.
这是在国画上训练的微调Stable Diffusion模型。
Use **guohua style** in your prompts for the effect.

## 示例图片
![example1](Untitled-1.png)
![example2](Untitled-3.png)

## 如何使用
```python
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2

pipe = pipeline(task=Tasks.text_to_image_synthesis, 
                model='langboat/Guohua-Diffusion',
                model_revision='v1.0')

prompt = 'The Godfather poster in guohua style'
output = pipe({'text': prompt})
cv2.imwrite('result.png', output['output_imgs'][0])

```

#### Diffusers
该模型可以像任何其他Stable Diffusion模型一样使用。
This model can be used just like any other Stable Diffusion model. 