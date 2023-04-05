
# **Mo Di Diffusion**

本模型自 Stable Diffusion 1.5 微调而来，微调数据来自某著名动画工作室的电影截图。在 prompt 中加入 `modern disney style` 可以在生成图像中实现该效果。



**一些生成效果图：**

![](modi-samples-01s.jpg)

![](modi-samples-02s.jpg)

![](modi-samples-03s.jpg)



**Lara Croft对应的 prompt 和设置：**

**modern disney lara croft**

*Steps: 50, Sampler: Euler a, CFG scale: 7, Seed: 3940025417, Size: 512x768*



**狮子王对应的 prompt 和设置：**

**modern disney (baby lion) **

**Negative prompt: person human** 

*Steps: 50, Sampler: Euler a, CFG scale: 7, Seed: 1355059992, Size: 512x512*



该模型使用diffusers自带的dreambooth样例代码（作者ShivamShrirao）训练9000个step而来。训练过程中使用了prior-preservation loss 和 *train-text-encoder* 选项。



## 使用方法：

```python
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2

pipe = pipeline(task=Tasks.text_to_image_synthesis, 
                model='dienstag/mo-di-diffusion',
                model_revision='v1.0.1')

prompt = 'a magical princess with golden hair, modern disney style'
output = pipe({'text': prompt})
cv2.imwrite('result.png', output['output_imgs'][0])
```



## License

  这个模型是开放的，所有人都可以使用，CreativeML OpenRAIL-M许可证进一步规定了权利和使用。CreativeML OpenRAIL许可证规定了：

  1. 你不能使用这个模型来故意产生或分享非法或有害的输出或内容。
  2. 作者对你产生的输出结果没有任何权利，你可以自由使用它们，并对它们的使用负责，但不得违反许可证中的规定。
  3. 你可以重新发布模型权重，并在商业上作为一项服务使用该模型。如果你这样做，请注意你必须包括与许可证中相同的使用限制，并将CreativeML OpenRAIL-M的副本分享给你的所有用户（请完全仔细阅读许可证）。 [请在这里阅读完整的许可证](https://huggingface.co/spaces/CompVis/stable-diffusion-license)。
